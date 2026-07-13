#!/usr/bin/env python3
r"""Uniform-iMPS linear-response solver for the 1D DLBP process.

The continuum is discretized into binary cells of width ``a``.  The state is a
translation-invariant nonnegative matrix-product state

    P(n_1,...,n_N) \propto tr[T[n_1] ... T[n_N]],  n_i in {0,1}.

The tensors are trained by minimizing the exact stationary master-equation
residual on a periodic collocation ring.  Observables are then evaluated in the
thermodynamic limit from the dominant fixed points of T[0] + T[1].

The derivative at d=0 is obtained by continuation to several small positive d
values followed by a constrained polynomial fit whose intercept is fixed to the
exact d=0 product measure.

This is a systematic finite-bond / finite-window approximation, not a moment
closure.  Convergence should be checked in:
    * MPS bond dimension D,
    * training-ring length N,
    * cell width a,
    * and the small-d fitting window.

An optional exact finite-ring Poisson-equation benchmark is included.

Dependencies: numpy, scipy, torch, matplotlib (optional for plotting).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F


Array = np.ndarray


@dataclass(frozen=True)
class ModelParameters:
    b: float = 1.0
    gamma: float = 1.0
    lam: float = 1.0
    a: float = 0.2

    @property
    def kappa(self) -> float:
        return 0.5 * self.lam

    @property
    def lattice_fugacity(self) -> float:
        # Exact hard-core product-state odds at d=0.
        return self.b * self.a / self.gamma

    @property
    def p0(self) -> float:
        z = self.lattice_fugacity
        return z / (1.0 + z)

    @property
    def rho0_lattice(self) -> float:
        return self.p0 / self.a


@dataclass
class TrainingData:
    bits_np: Array
    fields_np: Array
    flip_np: Array
    bits: torch.Tensor
    fields: torch.Tensor
    flip: torch.Tensor
    counts_np: Array
    counts: torch.Tensor


@dataclass
class FitResult:
    d: float
    loss: float
    tensors: Array  # shape (2, D, D)
    rho: float
    p1: float
    g2: Dict[int, float]
    k2: Dict[int, float]
    k3_connected: Dict[Tuple[int, int], float]
    k3_connected_normalized: Dict[Tuple[int, int], float]
    k3: Dict[Tuple[int, int], float]


def softplus_inverse(x: Array) -> Array:
    """Stable inverse of softplus for strictly positive x."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    large = x > 20.0
    out[large] = x[large]
    out[~large] = np.log(np.expm1(x[~large]))
    return out


def periodic_kernel_matrix(n_sites: int, params: ModelParameters) -> Array:
    """K_ij for a periodic collocation ring, excluding self-interaction."""
    idx = np.arange(n_sites)
    delta = np.abs(idx[:, None] - idx[None, :])
    delta = np.minimum(delta, n_sites - delta)
    distance = params.a * delta
    kernel = params.kappa * np.exp(-params.lam * distance)
    np.fill_diagonal(kernel, 0.0)
    return kernel


def enumerate_training_data(
    n_sites: int,
    params: ModelParameters,
    device: torch.device,
    dtype: torch.dtype,
) -> TrainingData:
    """Enumerate all binary configurations on the training ring."""
    if n_sites > 22:
        raise ValueError(
            "Exact residual enumeration scales as 2**N; use N <= 22 or add "
            "Monte-Carlo collocation."
        )

    n_cfg = 1 << n_sites
    states = np.arange(n_cfg, dtype=np.uint64)
    bits = ((states[:, None] >> np.arange(n_sites, dtype=np.uint64)) & 1).astype(
        np.float64
    )

    kernel = periodic_kernel_matrix(n_sites, params)
    # fields[cfg, i] = sum_{j != i} K_ij n_j.
    fields = bits @ kernel.T

    flips = np.empty((n_cfg, n_sites), dtype=np.int64)
    for i in range(n_sites):
        flips[:, i] = (states ^ np.uint64(1 << i)).astype(np.int64)

    return TrainingData(
        bits_np=bits,
        fields_np=fields,
        flip_np=flips,
        bits=torch.as_tensor(bits, dtype=dtype, device=device),
        fields=torch.as_tensor(fields, dtype=dtype, device=device),
        flip=torch.as_tensor(flips, dtype=torch.long, device=device),
        counts_np=bits.sum(axis=1),
        counts=torch.as_tensor(bits.sum(axis=1), dtype=dtype, device=device),
    )


def product_initial_tensors(
    bond_dim: int,
    p: float,
    hidden_mixing: float = 0.2,
    perturbation: float = 1e-4,
    seed: int = 0,
) -> Array:
    """Positive tensors representing the exact product measure at d=0.

    T_0 = (1-p) H and T_1 = p H, where H is a primitive hidden transfer
    matrix.  Their physical output is exactly Bernoulli even for D > 1.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1)")
    if bond_dim < 1:
        raise ValueError("bond_dim must be positive")

    eye = np.eye(bond_dim)
    all_to_all = np.ones((bond_dim, bond_dim)) / bond_dim
    hidden = (1.0 - hidden_mixing) * eye + hidden_mixing * all_to_all
    tensors = np.stack([(1.0 - p) * hidden, p * hidden], axis=0)

    if perturbation > 0.0 and bond_dim > 1:
        rng = np.random.default_rng(seed)
        tensors *= np.exp(perturbation * rng.standard_normal(tensors.shape))
    return tensors


class UniformPositiveMPS(torch.nn.Module):
    """Translation-invariant nonnegative periodic MPS with two physical states."""

    def __init__(self, initial_tensors: Array, dtype: torch.dtype, device: torch.device):
        super().__init__()
        if initial_tensors.ndim != 3 or initial_tensors.shape[0] != 2:
            raise ValueError("initial_tensors must have shape (2, D, D)")
        raw = softplus_inverse(np.maximum(initial_tensors, 1e-12))
        self.raw = torch.nn.Parameter(torch.as_tensor(raw, dtype=dtype, device=device))

    def positive_tensors(self) -> torch.Tensor:
        tensors = F.softplus(self.raw) + 1e-14
        # Divide by a common scalar.  This is an exact gauge rescaling because
        # every length-N configuration acquires the same factor.
        scale = tensors.sum()
        return tensors / scale

    def probabilities(self, bits: torch.Tensor) -> torch.Tensor:
        """Exact periodic-ring probabilities for all enumerated bit strings."""
        tensors = self.positive_tensors()
        t0, t1 = tensors[0], tensors[1]
        n_cfg, n_sites = bits.shape
        bond_dim = t0.shape[0]

        prod = torch.eye(bond_dim, dtype=bits.dtype, device=bits.device)
        prod = prod.unsqueeze(0).expand(n_cfg, -1, -1).clone()
        for i in range(n_sites):
            selector = bits[:, i].reshape(n_cfg, 1, 1)
            local = t0.unsqueeze(0) + selector * (t1 - t0).unsqueeze(0)
            prod = torch.bmm(prod, local)
        weights = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1)
        weights = torch.clamp(weights, min=1e-300)
        return weights / weights.sum()


def stationarity_residual(
    probabilities: torch.Tensor,
    data: TrainingData,
    params: ModelParameters,
    d: float,
    reflect_vacuum: bool = True,
) -> torch.Tensor:
    """Q(d) P for the binary-cell DLBP generator.

    For a target configuration x and site i:
      * if x_i=1, incoming flux is a birth from x with site i emptied;
      * if x_i=0, incoming flux is a death from x with site i occupied.
    The exterior field excludes site i and is unchanged by flipping n_i.
    """
    birth_rate = params.b * params.a * data.fields
    death_incoming = d + params.gamma * data.fields
    death_outgoing = death_incoming
    if reflect_vacuum:
        # Active/quasi-stationary regularization: suppress only the intrinsic
        # death transition from a singleton to the vacuum.  In the
        # thermodynamic limit its effect is exponentially small.
        singleton = (data.counts == 1.0)[:, None]
        death_outgoing = torch.where(singleton, params.gamma * data.fields, death_outgoing)

    p_flip = probabilities[data.flip]
    p_here = probabilities[:, None]
    occupied = data.bits > 0.5

    incoming_rate = torch.where(occupied, birth_rate, death_incoming)
    outgoing_rate = torch.where(occupied, death_outgoing, birth_rate)
    return (incoming_rate * p_flip - outgoing_rate * p_here).sum(dim=1)


def stationary_loss(
    model: UniformPositiveMPS,
    data: TrainingData,
    params: ModelParameters,
    d: float,
    probability_floor: float = 1e-14,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probabilities = model.probabilities(data.bits)
    # Condition the finite collocation ring on the active (nonempty) sector.
    # This removes the vacuum stationary solution while leaving the infinite
    # positive-density limit unchanged.
    probabilities = probabilities.clone()
    probabilities[0] = 0.0
    probabilities = probabilities / probabilities.sum()
    residual = stationarity_residual(probabilities, data, params, d, reflect_vacuum=True)
    residual = residual[1:]
    probabilities_active = probabilities[1:]

    # Pearson/chi-square residual: treats rare and common configurations on a
    # comparable relative scale.  The residual sums to zero automatically.
    denom = probabilities_active + probability_floor
    loss = torch.sum(residual.square() / denom)
    return loss, probabilities


def optimize_for_d(
    model: UniformPositiveMPS,
    data: TrainingData,
    params: ModelParameters,
    d: float,
    adam_steps: int,
    adam_lr: float,
    lbfgs_steps: int,
    verbose: bool,
) -> float:
    """Continuation solve for one mortality value."""
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    best_loss = math.inf
    best_raw: Optional[torch.Tensor] = None

    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = stationary_loss(model, data, params, d)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss during Adam optimization")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()

        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_raw = model.raw.detach().clone()
        if verbose and (step % max(1, adam_steps // 10) == 0 or step == adam_steps - 1):
            print(f"  Adam {step:5d}: loss={value:.6e}")

    if best_raw is not None:
        with torch.no_grad():
            model.raw.copy_(best_raw)

    if lbfgs_steps > 0:
        optimizer2 = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=lbfgs_steps,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer2.zero_grad(set_to_none=True)
            loss, _ = stationary_loss(model, data, params, d)
            loss.backward()
            return loss

        optimizer2.step(closure)

    final_loss, _ = stationary_loss(model, data, params, d)
    return float(final_loss.detach().cpu())


def dominant_fixed_points(t: Array) -> Tuple[float, Array, Array]:
    """Perron eigenvalue and normalized left/right eigenvectors."""
    vals_r, vecs_r = np.linalg.eig(t)
    idx_r = int(np.argmax(vals_r.real))
    eig = float(vals_r[idx_r].real)
    r = np.real(vecs_r[:, idx_r])

    vals_l, vecs_l = np.linalg.eig(t.T)
    idx_l = int(np.argmin(np.abs(vals_l - eig)))
    l = np.real(vecs_l[:, idx_l])

    # Perron vectors can be returned with either sign.
    if np.sum(r) < 0:
        r = -r
    if np.sum(l) < 0:
        l = -l
    r = np.maximum(r, 0.0)
    l = np.maximum(l, 0.0)

    overlap = float(l @ r)
    if overlap <= 0.0:
        raise RuntimeError("failed to obtain positive Perron fixed points")
    l /= overlap
    return eig, l, r


def matrix_power(t: Array, n: int) -> Array:
    if n < 0:
        raise ValueError("matrix power must be nonnegative")
    return np.linalg.matrix_power(t, n)


def infinite_mps_observables(
    tensors: Array,
    params: ModelParameters,
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """Thermodynamic-limit observables from the uniform transfer matrix."""
    t0, t1 = np.asarray(tensors[0]), np.asarray(tensors[1])
    transfer = t0 + t1
    eig, l, r = dominant_fixed_points(transfer)
    t0 = t0 / eig
    t1 = t1 / eig
    transfer = transfer / eig

    norm = float(l @ r)
    if abs(norm - 1.0) > 1e-8:
        l = l / norm

    p1 = float(l @ t1 @ r)
    rho = p1 / params.a

    p11: Dict[int, float] = {}
    k2: Dict[int, float] = {}
    g2: Dict[int, float] = {}
    for m in sorted(set(pair_separations)):
        if m < 1:
            raise ValueError("pair separations must be >= 1 lattice cell")
        value = float(l @ t1 @ matrix_power(transfer, m - 1) @ t1 @ r)
        p11[m] = value
        k2[m] = value / (params.a**2)
        g2[m] = value / (p1**2)

    # Cache any pair separations needed by connected triplets.
    needed_pair = set(pair_separations)
    for m, n in triplets:
        if not (1 <= m < n):
            raise ValueError("triplets must be specified as 1 <= m < n")
        needed_pair.update((m, n, n - m))
    for sep in sorted(needed_pair):
        if sep not in p11:
            value = float(l @ t1 @ matrix_power(transfer, sep - 1) @ t1 @ r)
            p11[sep] = value
            k2[sep] = value / (params.a**2)
            g2[sep] = value / (p1**2)

    k3: Dict[Tuple[int, int], float] = {}
    k3c: Dict[Tuple[int, int], float] = {}
    k3c_norm: Dict[Tuple[int, int], float] = {}
    for m, n in triplets:
        p111 = float(
            l
            @ t1
            @ matrix_power(transfer, m - 1)
            @ t1
            @ matrix_power(transfer, n - m - 1)
            @ t1
            @ r
        )
        k3_val = p111 / (params.a**3)
        connected = (
            k3_val
            - rho * (k2[m] + k2[n] + k2[n - m])
            + 2.0 * rho**3
        )
        k3[(m, n)] = k3_val
        k3c[(m, n)] = connected
        k3c_norm[(m, n)] = connected / (rho**3)

    return {
        "p1": p1,
        "rho": rho,
        "g2": g2,
        "k2": k2,
        "k3": k3,
        "k3_connected": k3c,
        "k3_connected_normalized": k3c_norm,
    }


def constrained_derivative_fit(
    d_values: Array,
    y_values: Array,
    y0: float,
    degree: int = 2,
) -> Tuple[float, Array]:
    """Fit y(d)=y0+c1*d+... with fixed intercept; return c1 and coefficients."""
    d_values = np.asarray(d_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if d_values.ndim != 1 or y_values.shape != d_values.shape:
        raise ValueError("d_values and y_values must be matching one-dimensional arrays")
    design = np.column_stack([d_values**k for k in range(1, degree + 1)])
    coeff, *_ = np.linalg.lstsq(design, y_values - y0, rcond=None)
    return float(coeff[0]), coeff


def run_imps_continuation(
    params: ModelParameters,
    bond_dim: int,
    n_train: int,
    d_values: Sequence[float],
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
    adam_steps: int,
    adam_lr: float,
    lbfgs_steps: int,
    seed: int,
    device_name: str,
    verbose: bool,
) -> List[FitResult]:
    dtype = torch.float64
    device = torch.device(device_name)
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = enumerate_training_data(n_train, params, device=device, dtype=dtype)
    initial = product_initial_tensors(
        bond_dim=bond_dim,
        p=params.p0,
        hidden_mixing=0.2,
        perturbation=1e-3 if bond_dim > 1 else 0.0,
        seed=seed,
    )
    model = UniformPositiveMPS(initial, dtype=dtype, device=device)

    results: List[FitResult] = []
    for d in sorted(d_values):
        if d <= 0.0:
            raise ValueError("continuation d-values must be strictly positive")
        if verbose:
            print(f"\nOptimizing d={d:.8g}, D={bond_dim}, N={n_train}")
        loss = optimize_for_d(
            model=model,
            data=data,
            params=params,
            d=d,
            adam_steps=adam_steps,
            adam_lr=adam_lr,
            lbfgs_steps=lbfgs_steps,
            verbose=verbose,
        )
        tensors = model.positive_tensors().detach().cpu().numpy()
        obs = infinite_mps_observables(
            tensors=tensors,
            params=params,
            pair_separations=pair_separations,
            triplets=triplets,
        )
        result = FitResult(
            d=float(d),
            loss=loss,
            tensors=tensors.copy(),
            rho=float(obs["rho"]),
            p1=float(obs["p1"]),
            g2=dict(obs["g2"]),
            k2=dict(obs["k2"]),
            k3_connected=dict(obs["k3_connected"]),
            k3_connected_normalized=dict(obs["k3_connected_normalized"]),
            k3=dict(obs["k3"]),
        )
        results.append(result)
        if verbose:
            print(f"  final loss={loss:.6e}, rho={result.rho:.10f}")
    return results


def exact_product_probability(bits: Array, p: float) -> Array:
    n = bits.sum(axis=1)
    n_sites = bits.shape[1]
    logp = n * math.log(p) + (n_sites - n) * math.log1p(-p)
    prob = np.exp(logp)
    return prob / prob.sum()


def generator_action_numpy(
    vector: Array,
    data: TrainingData,
    params: ModelParameters,
    d: float,
    reflect_vacuum: bool = True,
) -> Array:
    bits = data.bits_np.astype(bool)
    fields = data.fields_np
    birth = params.b * params.a * fields
    death_incoming = d + params.gamma * fields
    death_outgoing = death_incoming.copy()
    if reflect_vacuum:
        singleton = (data.counts_np == 1.0)[:, None]
        death_outgoing = np.where(singleton, params.gamma * fields, death_outgoing)
    vflip = vector[data.flip_np]
    incoming = np.where(bits, birth, death_incoming)
    outgoing = np.where(bits, death_outgoing, birth)
    return np.sum(incoming * vflip - outgoing * vector[:, None], axis=1)


def exact_finite_ring_response(
    data: TrainingData,
    params: ModelParameters,
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
    gmres_rtol: float = 1e-10,
    gmres_maxiter: int = 2000,
) -> Dict[str, object]:
    """Exact active-sector finite-ring Poisson-equation benchmark at d=0.

    The finite ring is conditioned to be nonempty and singleton intrinsic death
    is reflected.  This is the standard finite-volume regularization of the
    infinite-volume active branch.  We solve

        (Q0 + pi 1^T) p' = -Q1 pi,    sum p' = 0

    on the nonempty sector.
    """
    bits = data.bits_np
    n_cfg, n_sites = bits.shape
    pi_full = exact_product_probability(bits, params.p0)
    pi_full[0] = 0.0
    pi_full /= pi_full.sum()
    pi = pi_full[1:]

    q1pi_full = generator_action_numpy(
        pi_full, data, params, d=1.0, reflect_vacuum=True
    ) - generator_action_numpy(
        pi_full, data, params, d=0.0, reflect_vacuum=True
    )
    rhs = -q1pi_full[1:]

    def matvec(v: Array) -> Array:
        full = np.zeros(n_cfg, dtype=np.float64)
        full[1:] = v
        qv = generator_action_numpy(
            full, data, params, d=0.0, reflect_vacuum=True
        )[1:]
        return qv + pi * np.sum(v)

    operator = spla.LinearOperator(
        (n_cfg - 1, n_cfg - 1), matvec=matvec, dtype=np.float64
    )
    pprime_active, info = spla.gmres(
        operator,
        rhs,
        x0=np.zeros_like(rhs),
        rtol=gmres_rtol,
        atol=0.0,
        maxiter=gmres_maxiter,
    )
    if info != 0:
        raise RuntimeError(f"GMRES failed with info={info}")

    pprime = np.zeros(n_cfg, dtype=np.float64)
    pprime[1:] = pprime_active

    n_total = bits.sum(axis=1)
    rho_obs = n_total / (n_sites * params.a)
    rho0 = float(rho_obs @ pi_full)
    rho_prime = float(rho_obs @ pprime)

    pair_baseline: Dict[int, float] = {}
    pair_prime: Dict[int, float] = {}
    pair_out: Dict[int, Dict[str, float]] = {}
    needed_pair = set(pair_separations)
    for m, n in triplets:
        needed_pair.update((m, n, n - m))

    for sep in sorted(needed_pair):
        pair_count = np.zeros(n_cfg)
        for i in range(n_sites):
            pair_count += bits[:, i] * bits[:, (i + sep) % n_sites]
        k2_obs = pair_count / (n_sites * params.a**2)
        k20 = float(k2_obs @ pi_full)
        k2p = float(k2_obs @ pprime)
        pair_baseline[sep] = k20
        pair_prime[sep] = k2p
        if sep in pair_separations:
            g20 = k20 / rho0**2
            g2p = k2p / rho0**2 - 2.0 * k20 * rho_prime / rho0**3
            pair_out[sep] = {
                "k2_zero": k20,
                "g2_zero": g20,
                "k2_prime": k2p,
                "g2_prime": g2p,
            }

    triplet_out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for m, n in triplets:
        triple_count = np.zeros(n_cfg)
        for i in range(n_sites):
            triple_count += (
                bits[:, i]
                * bits[:, (i + m) % n_sites]
                * bits[:, (i + n) % n_sites]
            )
        k3_obs = triple_count / (n_sites * params.a**3)
        k30 = float(k3_obs @ pi_full)
        k3p = float(k3_obs @ pprime)

        k2_zeros = [pair_baseline[x] for x in (m, n, n - m)]
        k2_primes = [pair_prime[x] for x in (m, n, n - m)]
        c30 = k30 - rho0 * sum(k2_zeros) + 2.0 * rho0**3
        c3p = (
            k3p
            - rho_prime * sum(k2_zeros)
            - rho0 * sum(k2_primes)
            + 6.0 * rho0**2 * rho_prime
        )
        c3n0 = c30 / rho0**3
        c3np = c3p / rho0**3 - 3.0 * c30 * rho_prime / rho0**4
        triplet_out[(m, n)] = {
            "k3_zero": k30,
            "connected_k3_zero": c30,
            "normalized_connected_k3_zero": c3n0,
            "k3_prime": k3p,
            "connected_k3_prime": c3p,
            "normalized_connected_k3_prime": c3np,
        }

    return {
        "rho_zero": rho0,
        "rho_prime": rho_prime,
        "pairs": pair_out,
        "triplets": triplet_out,
        "normalization_error": float(abs(np.sum(pprime_active))),
        "linear_residual_norm": float(np.linalg.norm(matvec(pprime_active) - rhs)),
    }

def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def parse_triplets(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(x.strip()) for x in item.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "triplets must look like '1,2;1,3;2,4'"
            )
        m, n = parts
        if not (1 <= m < n):
            raise argparse.ArgumentTypeError("each triplet must satisfy 1 <= m < n")
        out.append((m, n))
    if not out:
        raise argparse.ArgumentTypeError("no triplets were supplied")
    return out



def json_safe(value):
    """Recursively convert tuple dictionary keys and NumPy scalars for JSON."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key = ",".join(str(x) for x in key)
            else:
                key = str(key) if not isinstance(key, (str, int, float, bool)) else key
            out[key] = json_safe(item)
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def write_outputs(
    output_dir: pathlib.Path,
    params: ModelParameters,
    bond_dim: int,
    n_train: int,
    fit_degree: int,
    results: Sequence[FitResult],
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
    exact_benchmark: Optional[Dict[str, object]],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d_values = np.array([x.d for x in results], dtype=np.float64)

    rho_values = np.array([x.rho for x in results])
    rho_prime, rho_coeff = constrained_derivative_fit(
        d_values, rho_values, params.rho0_lattice, degree=fit_degree
    )

    pair_derivatives: Dict[str, Dict[str, float]] = {}
    for m in pair_separations:
        g_values = np.array([x.g2[m] for x in results])
        k_values = np.array([x.k2[m] for x in results])
        g_prime, g_coeff = constrained_derivative_fit(
            d_values, g_values, 1.0, degree=fit_degree
        )
        k_prime, k_coeff = constrained_derivative_fit(
            d_values,
            k_values,
            params.rho0_lattice**2,
            degree=fit_degree,
        )
        pair_derivatives[str(m)] = {
            "distance": m * params.a,
            "g2_prime": g_prime,
            "k2_prime": k_prime,
            "g2_fit_coefficients": g_coeff.tolist(),
            "k2_fit_coefficients": k_coeff.tolist(),
        }

    triplet_derivatives: Dict[str, Dict[str, float]] = {}
    for key in triplets:
        c_values = np.array([x.k3_connected[key] for x in results])
        cn_values = np.array([x.k3_connected_normalized[key] for x in results])
        k3_values = np.array([x.k3[key] for x in results])
        c_prime, c_coeff = constrained_derivative_fit(
            d_values, c_values, 0.0, degree=fit_degree
        )
        cn_prime, cn_coeff = constrained_derivative_fit(
            d_values, cn_values, 0.0, degree=fit_degree
        )
        k3_prime, k3_coeff = constrained_derivative_fit(
            d_values,
            k3_values,
            params.rho0_lattice**3,
            degree=fit_degree,
        )
        label = f"{key[0]},{key[1]}"
        triplet_derivatives[label] = {
            "distances": [key[0] * params.a, key[1] * params.a],
            "connected_k3_prime": c_prime,
            "normalized_connected_k3_prime": cn_prime,
            "k3_prime": k3_prime,
            "connected_fit_coefficients": c_coeff.tolist(),
            "normalized_connected_fit_coefficients": cn_coeff.tolist(),
            "k3_fit_coefficients": k3_coeff.tolist(),
        }

    summary: Dict[str, object] = {
        "model": {
            "b": params.b,
            "gamma": params.gamma,
            "lambda": params.lam,
            "cell_width": params.a,
            "lattice_p0": params.p0,
            "lattice_rho0": params.rho0_lattice,
        },
        "ansatz": {
            "bond_dimension": bond_dim,
            "training_ring_sites": n_train,
            "training_ring_length": n_train * params.a,
            "fit_degree": fit_degree,
            "d_values": d_values.tolist(),
            "stationarity_losses": [x.loss for x in results],
        },
        "rho_prime": rho_prime,
        "rho_fit_coefficients": rho_coeff.tolist(),
        "pair_derivatives": pair_derivatives,
        "triplet_derivatives": triplet_derivatives,
        "exact_finite_ring_benchmark": json_safe(exact_benchmark),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "continuation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["d", "loss", "rho"]
        header += [f"g2_m{m}" for m in pair_separations]
        header += [f"k2_m{m}" for m in pair_separations]
        header += [f"k3c_{m}_{n}" for m, n in triplets]
        header += [f"k3c_norm_{m}_{n}" for m, n in triplets]
        writer.writerow(header)
        for x in results:
            row: List[float] = [x.d, x.loss, x.rho]
            row += [x.g2[m] for m in pair_separations]
            row += [x.k2[m] for m in pair_separations]
            row += [x.k3_connected[(m, n)] for m, n in triplets]
            row += [x.k3_connected_normalized[(m, n)] for m, n in triplets]
            writer.writerow(row)

    np.savez_compressed(
        output_dir / "tensors_and_data.npz",
        d_values=d_values,
        tensors=np.stack([x.tensors for x in results]),
        rho_values=rho_values,
        losses=np.array([x.loss for x in results]),
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(d_values, rho_values, "o", label="iMPS continuation")
        grid = np.linspace(0.0, float(np.max(d_values)), 200)
        fit = params.rho0_lattice + sum(
            rho_coeff[k - 1] * grid**k for k in range(1, fit_degree + 1)
        )
        plt.plot(grid, fit, label="constrained fit")
        plt.xlabel("intrinsic death d")
        plt.ylabel("mean density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "density_fit.png", dpi=180)
        plt.close()

        plt.figure()
        distances = np.array([m * params.a for m in pair_separations])
        derivatives = np.array(
            [pair_derivatives[str(m)]["g2_prime"] for m in pair_separations]
        )
        plt.plot(distances, derivatives, "o-")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("separation r")
        plt.ylabel("d g2(r) / d d at d=0")
        plt.tight_layout()
        plt.savefig(output_dir / "pair_response.png", dpi=180)
        plt.close()
    except Exception as exc:  # plotting is optional
        print(f"Plotting skipped: {exc}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.25, help="cell width dx")
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument(
        "--d-values",
        type=parse_float_list,
        default=parse_float_list("0.0025,0.005,0.01,0.02"),
    )
    parser.add_argument(
        "--pairs",
        type=parse_int_list,
        default=parse_int_list("1,2,3,4,5,6"),
        help="pair separations in lattice cells",
    )
    parser.add_argument(
        "--triplets",
        type=parse_triplets,
        default=parse_triplets("1,2;1,3;2,4"),
        help="semicolon-separated pairs m,n for sites 0,m,n",
    )
    parser.add_argument("--fit-degree", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=1200)
    parser.add_argument("--adam-lr", type=float, default=0.03)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("imps_output"))
    parser.add_argument(
        "--exact-benchmark",
        action="store_true",
        help="also solve the exact finite-ring linear-response equation",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    params = ModelParameters(b=args.b, gamma=args.gamma, lam=args.lam, a=args.a)

    if max(args.pairs + [n for pair in args.triplets for n in pair]) >= args.n_train // 2:
        print(
            "Warning: requested separations approach half the training-ring size; "
            "increase --n-train to reduce wraparound effects."
        )
    if args.n_train * args.a < 6.0 / args.lam:
        print(
            "Warning: training-ring length is less than six kernel decay lengths; "
            "finite-window effects may be appreciable."
        )

    results = run_imps_continuation(
        params=params,
        bond_dim=args.bond_dim,
        n_train=args.n_train,
        d_values=args.d_values,
        pair_separations=args.pairs,
        triplets=args.triplets,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
        seed=args.seed,
        device_name=args.device,
        verbose=not args.quiet,
    )

    exact: Optional[Dict[str, object]] = None
    if args.exact_benchmark:
        device = torch.device(args.device)
        data = enumerate_training_data(
            args.n_train, params, device=device, dtype=torch.float64
        )
        exact = exact_finite_ring_response(
            data=data,
            params=params,
            pair_separations=args.pairs,
            triplets=args.triplets,
        )

    summary = write_outputs(
        output_dir=args.output,
        params=params,
        bond_dim=args.bond_dim,
        n_train=args.n_train,
        fit_degree=args.fit_degree,
        results=results,
        pair_separations=args.pairs,
        triplets=args.triplets,
        exact_benchmark=exact,
    )

    print("\nLinear-response summary")
    print(f"  rho'(0) = {summary['rho_prime']:.12g}")
    print("  pair g2 derivatives:")
    for m, item in summary["pair_derivatives"].items():
        print(
            f"    m={m:>3s}, r={item['distance']:.6g}: "
            f"g2'(0)={item['g2_prime']:.12g}"
        )
    print("  connected triplet derivatives:")
    for label, item in summary["triplet_derivatives"].items():
        print(
            f"    (m,n)=({label}): "
            f"kappa3'(0)={item['connected_k3_prime']:.12g}"
        )
    print(f"\nWrote results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
