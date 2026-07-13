#!/usr/bin/env python3
r"""Thermodynamic-limit iMPS linear-response solver for the 1D DLBP process.

This version replaces the finite periodic training ring used by the earlier
prototype with a genuine one-site, translation-invariant infinite matrix
product distribution (classical iMPS / hidden-Markov MPS).  The tensor is put
in a stochastic right-canonical gauge,

    A[s] >= 0,        sum_s A[s] @ 1 = 1,

and its left Perron vector pi is the infinite left environment.  Cylinder
probabilities and all contributions from the two infinite exterior half-lines
are contracted exactly.  For the exponential kernel the exterior sums are
closed by the resolvent

    R_q = (I - q E)^(-1),      E = A[0] + A[1],

where q = exp(-lambda * dx).

The variational equation is an infinite-volume Galerkin/VUMPS-style projected
fixed-point equation: the exact stationary master-equation residual is required
to vanish for every binary word of a chosen projection length ell.  Increasing
ell and the MPS bond dimension D gives a systematic sequence; no periodic ring
or finite-volume vacuum conditioning is used.

The active branch is followed from the exact d=0 Bernoulli/PPP tensor to small
positive d.  Derivatives at d=0 are obtained from constrained polynomial fits.

This is inspired by the canonical InfiniteMPS + fixed-environment philosophy of
MPSKit/VUMPS, but it is adapted to a classical non-Hermitian Markov generator:
the left physical eigenvector is the flat summation state, so stochastic
canonicalization and cylinder flux residuals replace a Hermitian ground-state
Rayleigh quotient.

Dependencies: numpy, scipy, torch, matplotlib (optional).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

Array = np.ndarray


@dataclass(frozen=True)
class ModelParameters:
    b: float = 1.0
    gamma: float = 1.0
    lam: float = 1.0
    a: float = 0.25

    @property
    def kappa(self) -> float:
        return 0.5 * self.lam

    @property
    def q(self) -> float:
        return math.exp(-self.lam * self.a)

    @property
    def lattice_fugacity(self) -> float:
        return self.b * self.a / self.gamma

    @property
    def p0(self) -> float:
        z = self.lattice_fugacity
        return z / (1.0 + z)

    @property
    def rho0_lattice(self) -> float:
        return self.p0 / self.a


@dataclass
class CylinderData:
    bits: torch.Tensor               # (2**ell, ell)
    flip: torch.Tensor               # (2**ell, ell)
    inside_field: torch.Tensor       # (2**ell, ell)
    ell: int


@dataclass
class FitResult:
    d: float
    loss: float
    projected_residual_rms: float
    density: float
    p1: float
    tensors: Array
    hidden_stationary: Array
    transfer_spectrum: Array
    g2: Dict[int, float]
    k2: Dict[int, float]
    k3: Dict[Tuple[int, int], float]
    k3_connected: Dict[Tuple[int, int], float]
    k3_connected_normalized: Dict[Tuple[int, int], float]


def enumerate_cylinder(
    ell: int,
    params: ModelParameters,
    device: torch.device,
    dtype: torch.dtype,
) -> CylinderData:
    if ell < 1:
        raise ValueError("projection length ell must be positive")
    if ell > 18:
        raise ValueError(
            "The exact word projection uses 2**ell words; ell > 18 is usually "
            "impractical. Increase bond dimension before increasing ell further."
        )

    n_words = 1 << ell
    states = np.arange(n_words, dtype=np.uint64)
    bits_np = (
        (states[:, None] >> np.arange(ell, dtype=np.uint64)) & np.uint64(1)
    ).astype(np.float64)

    flip_np = np.empty((n_words, ell), dtype=np.int64)
    for i in range(ell):
        flip_np[:, i] = (states ^ np.uint64(1 << i)).astype(np.int64)

    idx = np.arange(ell)
    distances = np.abs(idx[:, None] - idx[None, :])
    kernel = params.kappa * np.exp(-params.lam * params.a * distances)
    np.fill_diagonal(kernel, 0.0)
    # H_inside[word, i] = sum_{j in cylinder, j != i} K_ij n_j.
    inside_np = bits_np @ kernel.T

    return CylinderData(
        bits=torch.as_tensor(bits_np, dtype=dtype, device=device),
        flip=torch.as_tensor(flip_np, dtype=torch.long, device=device),
        inside_field=torch.as_tensor(inside_np, dtype=dtype, device=device),
        ell=ell,
    )


def product_hidden_transition(bond_dim: int, mixing: float) -> Array:
    """Primitive row-stochastic hidden transfer used to embed the product state."""
    if bond_dim < 1:
        raise ValueError("bond dimension must be positive")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("hidden mixing must lie in (0,1]")
    eye = np.eye(bond_dim, dtype=np.float64)
    uniform = np.ones((bond_dim, bond_dim), dtype=np.float64) / bond_dim
    return (1.0 - mixing) * eye + mixing * uniform


def tensors_to_free_logits(tensors: Array) -> Array:
    """Inverse of the row-softmax gauge, fixing the last logit of each row to 0."""
    tensors = np.asarray(tensors, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[0] != 2 or tensors.shape[1] != tensors.shape[2]:
        raise ValueError("tensors must have shape (2,D,D)")
    D = tensors.shape[1]
    row_probs = tensors.transpose(1, 0, 2).reshape(D, 2 * D)
    row_probs = np.maximum(row_probs, 1e-300)
    row_probs /= row_probs.sum(axis=1, keepdims=True)
    logits = np.log(row_probs)
    return logits[:, :-1] - logits[:, -1:]


def product_initial_tensors(
    bond_dim: int,
    p1: float,
    hidden_mixing: float,
    perturbation: float,
    seed: int,
) -> Array:
    if not (0.0 < p1 < 1.0):
        raise ValueError("p1 must lie in (0,1)")
    H = product_hidden_transition(bond_dim, hidden_mixing)
    if perturbation > 0.0 and bond_dim > 1:
        # Perturb only the hidden transition.  Keeping A_s = p_s H preserves
        # the exact physical Bernoulli product measure at d=0.
        rng = np.random.default_rng(seed)
        H = H * np.exp(perturbation * rng.standard_normal(H.shape))
        H /= H.sum(axis=1, keepdims=True)
    return np.stack([(1.0 - p1) * H, p1 * H], axis=0)


class StochasticInfiniteMPS(torch.nn.Module):
    """One-site translation-invariant classical iMPS in right-canonical gauge.

    Each hidden-state row is a probability distribution over (physical state,
    next hidden state). Therefore E=A0+A1 is exactly row stochastic and the
    right fixed point is the all-ones vector.
    """

    def __init__(
        self,
        initial_tensors: Array,
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        super().__init__()
        free = tensors_to_free_logits(initial_tensors)
        self.free_logits = torch.nn.Parameter(
            torch.as_tensor(free, dtype=dtype, device=device)
        )

    @property
    def bond_dim(self) -> int:
        return int(self.free_logits.shape[0])

    def tensors_from_free(self, free_logits: torch.Tensor) -> torch.Tensor:
        D = free_logits.shape[0]
        zero = torch.zeros((D, 1), dtype=free_logits.dtype, device=free_logits.device)
        full = torch.cat([free_logits, zero], dim=1)
        rows = torch.softmax(full, dim=1)
        return rows.reshape(D, 2, D).permute(1, 0, 2).contiguous()

    def tensors(self) -> torch.Tensor:
        return self.tensors_from_free(self.free_logits)

    def left_fixed_point(self, tensors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalized left Perron vector of the row-stochastic transfer.

        We solve (E^T-I) pi = 0 with the final equation replaced by
        sum(pi)=1.  This is the exact one-site infinite environment and avoids
        differentiating through a long power iteration.
        """
        A = self.tensors() if tensors is None else tensors
        E = A[0] + A[1]
        D = E.shape[0]
        matrix = E.transpose(0, 1) - torch.eye(D, dtype=E.dtype, device=E.device)
        matrix = torch.cat([matrix[:-1], torch.ones((1, D), dtype=E.dtype, device=E.device)], dim=0)
        rhs = torch.zeros((D,), dtype=E.dtype, device=E.device)
        rhs[-1] = 1.0
        pi = torch.linalg.solve(matrix, rhs)
        return pi

    def canonical_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.tensors()
        E = A[0] + A[1]
        pi = self.left_fixed_point(A)
        return A, E, pi



def batch_word_products(A: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Return A[x0]...A[x_{ell-1}] for every binary word."""
    n_words, ell = bits.shape
    D = A.shape[1]
    prod = torch.eye(D, dtype=A.dtype, device=A.device)
    prod = prod.unsqueeze(0).expand(n_words, -1, -1).clone()
    A0 = A[0].unsqueeze(0)
    A1 = A[1].unsqueeze(0)
    for i in range(ell):
        x = bits[:, i].reshape(n_words, 1, 1)
        local = A0 + x * (A1 - A0)
        prod = torch.bmm(prod, local)
    return prod


def cylinder_probabilities_and_fields(
    A: torch.Tensor,
    E: torch.Tensor,
    pi: torch.Tensor,
    data: CylinderData,
    params: ModelParameters,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Cylinder probabilities and unnormalized field insertions.

    F[word,i] = E[ 1_{cylinder=word} H_i ], where H_i is the complete
    competitive/birth field at site i excluding site i itself. Contributions
    from both infinite half-lines are contracted exactly.
    """
    word_prod = batch_word_products(A, data.bits)
    D = E.shape[0]
    ones = torch.ones((D,), dtype=E.dtype, device=E.device)

    # P(word) = pi A_word 1.
    left_word = torch.einsum("a,wab->wb", pi, word_prod)
    probs = torch.einsum("wa,a->w", left_word, ones)

    q = torch.as_tensor(params.q, dtype=E.dtype, device=E.device)
    eye = torch.eye(D, dtype=E.dtype, device=E.device)
    resolvent = torch.linalg.solve(eye - q * E, eye)

    # Sum over particles at sites -1,-2,... to the left of the cylinder.
    left_boundary_row = pi @ A[1] @ resolvent
    left_base = params.kappa * torch.einsum(
        "a,wab,b->w", left_boundary_row, word_prod, ones
    )

    # Sum over particles at sites ell,ell+1,... to the right.
    right_boundary_col = resolvent @ A[1] @ ones
    right_base = params.kappa * torch.einsum(
        "a,wab,b->w", pi, word_prod, right_boundary_col
    )

    ell = data.ell
    site = torch.arange(ell, dtype=E.dtype, device=E.device)
    left_weights = q ** (site + 1.0)
    right_weights = q ** (ell - site)

    fields = probs[:, None] * data.inside_field
    fields = fields + left_base[:, None] * left_weights[None, :]
    fields = fields + right_base[:, None] * right_weights[None, :]
    return probs, fields


def projected_stationarity_residual(
    model: StochasticInfiniteMPS,
    data: CylinderData,
    params: ModelParameters,
    d: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""Exact infinite-volume flux residual for every cylinder word.

    Only flips inside the cylinder change its word. Their rates contain the
    complete infinite field, evaluated by cylinder_probabilities_and_fields.
    """
    A, E, pi = model.canonical_data()
    probs, fields = cylinder_probabilities_and_fields(A, E, pi, data, params)

    n_words, ell = data.bits.shape
    site_idx = torch.arange(ell, device=data.bits.device).reshape(1, ell)
    site_idx = site_idx.expand(n_words, -1)
    p_flip = probs[data.flip]
    f_flip = fields[data.flip, site_idx]

    occupied = data.bits > 0.5
    birth_here = params.b * params.a * fields
    birth_flip = params.b * params.a * f_flip
    death_here = d * probs[:, None] + params.gamma * fields
    death_flip = d * p_flip + params.gamma * f_flip

    incoming = torch.where(occupied, birth_flip, death_flip)
    outgoing = torch.where(occupied, death_here, birth_here)
    residual = (incoming - outgoing).sum(dim=1)

    ones = torch.ones((E.shape[0],), dtype=E.dtype, device=E.device)
    p1 = pi @ A[1] @ ones
    rho = p1 / params.a
    fp_error = torch.linalg.vector_norm(pi @ E - pi)
    row_error = torch.linalg.vector_norm(E @ ones - ones)
    diagnostics = {
        "probs": probs,
        "fields": fields,
        "p1": p1,
        "rho": rho,
        "pi": pi,
        "A": A,
        "E": E,
        "fp_error": fp_error,
        "row_error": row_error,
    }
    return residual, diagnostics


def loss_and_diagnostics(
    model: StochasticInfiniteMPS,
    data: CylinderData,
    params: ModelParameters,
    d: float,
    branch_fraction: float,
    branch_weight: float,
    probability_floor: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    residual, diag = projected_stationarity_residual(model, data, params, d)
    probs = diag["probs"]
    weighted = residual / torch.sqrt(probs + probability_floor)
    loss = torch.mean(weighted.square())

    # Select the active continuation branch. This barrier is exactly inactive
    # whenever rho stays above the requested fraction of the d=0 density.
    min_rho = branch_fraction * params.rho0_lattice
    barrier = torch.relu(
        torch.as_tensor(min_rho, dtype=loss.dtype, device=loss.device) - diag["rho"]
    )
    loss = loss + branch_weight * barrier.square()
    diag["weighted_residual"] = weighted
    diag["barrier"] = barrier
    return loss, diag


def optimize_adam_lbfgs(
    model: StochasticInfiniteMPS,
    data: CylinderData,
    params: ModelParameters,
    d: float,
    adam_steps: int,
    adam_lr: float,
    lbfgs_steps: int,
    branch_fraction: float,
    branch_weight: float,
    probability_floor: float,
    verbose: bool,
) -> Tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    best_loss = math.inf
    best_param: Optional[torch.Tensor] = None

    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, diag = loss_and_diagnostics(
            model, data, params, d,
            branch_fraction, branch_weight, probability_floor,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite iMPS loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_param = model.free_logits.detach().clone()
        if verbose and (step % max(1, adam_steps // 10) == 0 or step == adam_steps - 1):
            rms = float(torch.sqrt(torch.mean(diag["weighted_residual"].square())).detach().cpu())
            rho = float(diag["rho"].detach().cpu())
            print(f"  Adam {step:5d}: loss={value:.6e}, rms={rms:.6e}, rho={rho:.10f}")

    if best_param is not None:
        with torch.no_grad():
            model.free_logits.copy_(best_param)

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
            value, _ = loss_and_diagnostics(
                model, data, params, d,
                branch_fraction, branch_weight, probability_floor,
            )
            value.backward()
            return value

        optimizer2.step(closure)

    final_loss, diag = loss_and_diagnostics(
        model, data, params, d,
        branch_fraction, branch_weight, probability_floor,
    )
    rms = torch.sqrt(torch.mean(diag["weighted_residual"].square()))
    return float(final_loss.detach().cpu()), float(rms.detach().cpu())


def infinite_mps_observables(
    A: Array,
    pi: Array,
    params: ModelParameters,
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    A = np.asarray(A, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    E = A[0] + A[1]
    D = E.shape[0]
    one = np.ones(D)

    p1 = float(pi @ A[1] @ one)
    rho = p1 / params.a

    needed_pair = set(pair_separations)
    for m, n in triplets:
        needed_pair.update((m, n, n - m))

    p11: Dict[int, float] = {}
    k2: Dict[int, float] = {}
    g2: Dict[int, float] = {}
    for m in sorted(needed_pair):
        if m < 1:
            raise ValueError("pair separations must be >=1")
        value = float(pi @ A[1] @ np.linalg.matrix_power(E, m - 1) @ A[1] @ one)
        p11[m] = value
        k2[m] = value / params.a**2
        g2[m] = value / p1**2

    k3: Dict[Tuple[int, int], float] = {}
    k3c: Dict[Tuple[int, int], float] = {}
    k3cn: Dict[Tuple[int, int], float] = {}
    for m, n in triplets:
        if not (1 <= m < n):
            raise ValueError("triplets must satisfy 1 <= m < n")
        p111 = float(
            pi @ A[1]
            @ np.linalg.matrix_power(E, m - 1)
            @ A[1]
            @ np.linalg.matrix_power(E, n - m - 1)
            @ A[1]
            @ one
        )
        k3v = p111 / params.a**3
        conn = k3v - rho * (k2[m] + k2[n] + k2[n - m]) + 2.0 * rho**3
        k3[(m, n)] = k3v
        k3c[(m, n)] = conn
        k3cn[(m, n)] = conn / rho**3

    return {
        "p1": p1,
        "rho": rho,
        "g2": g2,
        "k2": k2,
        "k3": k3,
        "k3_connected": k3c,
        "k3_connected_normalized": k3cn,
    }


def transfer_spectrum(A: Array) -> Array:
    vals = np.linalg.eigvals(np.asarray(A[0]) + np.asarray(A[1]))
    order = np.argsort(-np.abs(vals))
    return vals[order]


def run_continuation(
    params: ModelParameters,
    bond_dim: int,
    projection_length: int,
    d_values: Sequence[float],
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
    hidden_mixing: float,
    perturbation: float,
    adam_steps: int,
    adam_lr: float,
    lbfgs_steps: int,
    branch_fraction: float,
    branch_weight: float,
    probability_floor: float,
    seed: int,
    device_name: str,
    verbose: bool,
) -> List[FitResult]:
    dtype = torch.float64
    device = torch.device(device_name)
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = enumerate_cylinder(projection_length, params, device, dtype)
    initial = product_initial_tensors(
        bond_dim=bond_dim,
        p1=params.p0,
        hidden_mixing=hidden_mixing,
        perturbation=perturbation if bond_dim > 1 else 0.0,
        seed=seed,
    )
    model = StochasticInfiniteMPS(
        initial_tensors=initial,
        dtype=dtype,
        device=device,
    )

    # Verify that the exact d=0 product measure is in the ansatz and satisfies
    # the infinite cylinder equations before continuation.
    with torch.no_grad():
        r0, diag0 = projected_stationarity_residual(model, data, params, d=0.0)
        d0_rms = float(
            torch.sqrt(torch.mean((r0 / torch.sqrt(diag0["probs"] + probability_floor)) ** 2))
            .detach().cpu()
        )
    if verbose:
        print(f"Exact d=0 infinite-cylinder residual RMS: {d0_rms:.6e}")

    results: List[FitResult] = []
    for d in sorted(d_values):
        if d <= 0.0:
            raise ValueError("all continuation d-values must be positive")
        if verbose:
            print(
                f"\nInfinite-iMPS solve: d={d:.8g}, D={bond_dim}, "
                f"projection ell={projection_length}"
            )
        loss, rms = optimize_adam_lbfgs(
            model=model,
            data=data,
            params=params,
            d=d,
            adam_steps=adam_steps,
            adam_lr=adam_lr,
            lbfgs_steps=lbfgs_steps,
            branch_fraction=branch_fraction,
            branch_weight=branch_weight,
            probability_floor=probability_floor,
            verbose=verbose,
        )
        with torch.no_grad():
            A_t, E_t, pi_t = model.canonical_data()
            A = A_t.detach().cpu().numpy()
            pi = pi_t.detach().cpu().numpy()
        obs = infinite_mps_observables(
            A, pi, params, pair_separations, triplets
        )
        spec = transfer_spectrum(A)
        result = FitResult(
            d=float(d),
            loss=loss,
            projected_residual_rms=rms,
            density=float(obs["rho"]),
            p1=float(obs["p1"]),
            tensors=A.copy(),
            hidden_stationary=pi.copy(),
            transfer_spectrum=spec.copy(),
            g2=dict(obs["g2"]),
            k2=dict(obs["k2"]),
            k3=dict(obs["k3"]),
            k3_connected=dict(obs["k3_connected"]),
            k3_connected_normalized=dict(obs["k3_connected_normalized"]),
        )
        results.append(result)
        if verbose:
            sub = spec[1] if len(spec) > 1 else 0.0
            print(
                f"  final loss={loss:.6e}, projected RMS={rms:.6e}, "
                f"rho={result.density:.10f}, lambda_2={sub}"
            )
    return results


def constrained_derivative_fit(
    d_values: Array,
    y_values: Array,
    y0: float,
    degree: int,
) -> Tuple[float, Array]:
    d_values = np.asarray(d_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if degree < 1:
        raise ValueError("fit degree must be >=1")
    design = np.column_stack([d_values**k for k in range(1, degree + 1)])
    coeff, *_ = np.linalg.lstsq(design, y_values - y0, rcond=None)
    return float(coeff[0]), coeff


def json_safe(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key = ",".join(str(x) for x in key)
            out[str(key)] = json_safe(item)
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [[float(z.real), float(z.imag)] for z in value.ravel()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def write_outputs(
    output_dir: pathlib.Path,
    params: ModelParameters,
    bond_dim: int,
    projection_length: int,
    fit_degree: int,
    results: Sequence[FitResult],
    pair_separations: Sequence[int],
    triplets: Sequence[Tuple[int, int]],
    max_residual_rms: float = 1e-6,
    max_loss: float = 1e-8,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    d_values = np.array([x.d for x in results], dtype=np.float64)
    rho_values = np.array([x.density for x in results], dtype=np.float64)
    rho_prime, rho_coeff = constrained_derivative_fit(
        d_values, rho_values, params.rho0_lattice, fit_degree
    )

    pair_derivatives: Dict[str, Dict[str, object]] = {}
    for m in pair_separations:
        g_values = np.array([x.g2[m] for x in results])
        k_values = np.array([x.k2[m] for x in results])
        gp, gc = constrained_derivative_fit(d_values, g_values, 1.0, fit_degree)
        kp, kc = constrained_derivative_fit(
            d_values, k_values, params.rho0_lattice**2, fit_degree
        )
        pair_derivatives[str(m)] = {
            "distance": m * params.a,
            "g2_prime": gp,
            "k2_prime": kp,
            "g2_fit_coefficients": gc.tolist(),
            "k2_fit_coefficients": kc.tolist(),
        }

    triplet_derivatives: Dict[str, Dict[str, object]] = {}
    for key in triplets:
        c = np.array([x.k3_connected[key] for x in results])
        cn = np.array([x.k3_connected_normalized[key] for x in results])
        k3 = np.array([x.k3[key] for x in results])
        cp, cc = constrained_derivative_fit(d_values, c, 0.0, fit_degree)
        cnp, cnc = constrained_derivative_fit(d_values, cn, 0.0, fit_degree)
        k3p, k3c = constrained_derivative_fit(
            d_values, k3, params.rho0_lattice**3, fit_degree
        )
        label = f"{key[0]},{key[1]}"
        triplet_derivatives[label] = {
            "distances": [key[0] * params.a, key[1] * params.a],
            "connected_k3_prime": cp,
            "normalized_connected_k3_prime": cnp,
            "k3_prime": k3p,
            "connected_fit_coefficients": cc.tolist(),
            "normalized_connected_fit_coefficients": cnc.tolist(),
            "k3_fit_coefficients": k3c.tolist(),
        }

    summary: Dict[str, object] = {
        "model": {
            "b": params.b,
            "gamma": params.gamma,
            "lambda": params.lam,
            "cell_width": params.a,
            "q": params.q,
            "lattice_p0": params.p0,
            "lattice_rho0": params.rho0_lattice,
            "kernel_type": "exponential",
            "kernel_variance": 2.0 / (params.lam ** 2),
            "kernel_normalization": "K(r)=(lambda/2)*exp(-lambda*abs(r))",
        },
        "method": {
            "name": "thermodynamic-limit stochastic iMPS cylinder projection",
            "unit_cell_sites": 1,
            "bond_dimension": bond_dim,
            "projection_length": projection_length,
            "number_of_projected_words": 1 << projection_length,
            "finite_periodic_ring": False,
            "exterior_half_lines": "exact transfer-resolvent contraction",
            "fit_degree": fit_degree,
            "d_values": d_values.tolist(),
            "losses": [x.loss for x in results],
            "projected_residual_rms": [x.projected_residual_rms for x in results],
            "max_projected_residual_rms": max((x.projected_residual_rms for x in results), default=float("nan")),
            "max_loss": max((x.loss for x in results), default=float("nan")),
            "residual_rms_threshold": max_residual_rms,
            "loss_threshold": max_loss,
            "diagnostics_ok": bool(
                all(x.projected_residual_rms <= max_residual_rms for x in results)
                and all(x.loss <= max_loss for x in results)
            ),
        },
        "rho_prime": rho_prime,
        "rho_fit_coefficients": rho_coeff.tolist(),
        "pair_derivatives": pair_derivatives,
        "triplet_derivatives": triplet_derivatives,
        "transfer_spectra": [json_safe(x.transfer_spectrum) for x in results],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    with (output_dir / "continuation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["d", "loss", "projected_residual_rms", "rho"]
        header += [f"g2_m{m}" for m in pair_separations]
        header += [f"k2_m{m}" for m in pair_separations]
        header += [f"k3c_{m}_{n}" for m, n in triplets]
        writer.writerow(header)
        for x in results:
            row: List[float] = [x.d, x.loss, x.projected_residual_rms, x.density]
            row += [x.g2[m] for m in pair_separations]
            row += [x.k2[m] for m in pair_separations]
            row += [x.k3_connected[(m, n)] for m, n in triplets]
            writer.writerow(row)

    np.savez_compressed(
        output_dir / "imps_tensors.npz",
        d_values=d_values,
        tensors=np.stack([x.tensors for x in results]),
        hidden_stationary=np.stack([x.hidden_stationary for x in results]),
        transfer_spectrum=np.stack([x.transfer_spectrum for x in results]),
        rho_values=rho_values,
        losses=np.array([x.loss for x in results]),
        projected_residual_rms=np.array([x.projected_residual_rms for x in results]),
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(d_values, rho_values, "o", label="infinite-iMPS continuation")
        grid = np.linspace(0.0, float(np.max(d_values)), 200)
        fit = params.rho0_lattice + sum(
            rho_coeff[k - 1] * grid**k for k in range(1, fit_degree + 1)
        )
        plt.plot(grid, fit, label="fixed-intercept fit")
        plt.xlabel("intrinsic death d")
        plt.ylabel("mean density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "density_fit.png", dpi=180)
        plt.close()

        plt.figure()
        distance = np.array([m * params.a for m in pair_separations])
        response = np.array([pair_derivatives[str(m)]["g2_prime"] for m in pair_separations])
        plt.plot(distance, response, "o-")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("separation r")
        plt.ylabel("d g2(r) / d d at d=0")
        plt.tight_layout()
        plt.savefig(output_dir / "pair_response.png", dpi=180)
        plt.close()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    return summary


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_triplets(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = [int(x.strip()) for x in item.split(",")]
        if len(values) != 2 or not (1 <= values[0] < values[1]):
            raise argparse.ArgumentTypeError(
                "triplets must look like '1,2;1,3;2,4' with 1 <= m < n"
            )
        out.append((values[0], values[1]))
    if not out:
        raise argparse.ArgumentTypeError("no triplets supplied")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.25, help="cell width dx")
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument(
        "--projection-length", type=int, default=7,
        help="number ell of consecutive sites whose 2**ell word fluxes are projected",
    )
    parser.add_argument(
        "--d-values", type=parse_float_list,
        default=parse_float_list("0.00125,0.0025,0.005,0.01"),
    )
    parser.add_argument(
        "--pairs", type=parse_int_list,
        default=parse_int_list("1,2,3,4,5,6,8,10"),
    )
    parser.add_argument(
        "--triplets", type=parse_triplets,
        default=parse_triplets("1,2;1,3;2,4;2,6"),
    )
    parser.add_argument("--fit-degree", type=int, default=2)
    parser.add_argument("--hidden-mixing", type=float, default=0.25)
    parser.add_argument("--initial-perturbation", type=float, default=1e-3)
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--adam-lr", type=float, default=0.02)
    parser.add_argument("--lbfgs-steps", type=int, default=50)
    parser.add_argument(
        "--branch-fraction", type=float, default=0.25,
        help="inactive lower-density barrier selecting the active branch",
    )
    parser.add_argument("--branch-weight", type=float, default=1e4)
    parser.add_argument("--probability-floor", type=float, default=1e-15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=pathlib.Path("imps_infinite_output"),
    )
    parser.add_argument("--max-residual-rms", type=float, default=1e-6)
    parser.add_argument("--max-loss", type=float, default=1e-8)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = ModelParameters(b=args.b, gamma=args.gamma, lam=args.lam, a=args.a)
    torch.set_num_threads(max(1, args.torch_threads))

    max_sep = max(args.pairs + [x for pair in args.triplets for x in pair])
    if max_sep < 1:
        raise ValueError("all requested separations must be positive")
    if args.projection_length < 2:
        print("Warning: projection length <2 cannot directly constrain pair fluxes.")
    if args.a * max_sep < 0.5 / args.lam:
        print(
            "Note: all requested observables are at short distances. Include larger "
            "separations to test the transfer-matrix tail."
        )

    results = run_continuation(
        params=params,
        bond_dim=args.bond_dim,
        projection_length=args.projection_length,
        d_values=args.d_values,
        pair_separations=args.pairs,
        triplets=args.triplets,
        hidden_mixing=args.hidden_mixing,
        perturbation=args.initial_perturbation,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
        branch_fraction=args.branch_fraction,
        branch_weight=args.branch_weight,
        probability_floor=args.probability_floor,
        seed=args.seed,
        device_name=args.device,
        verbose=not args.quiet,
    )

    summary = write_outputs(
        output_dir=args.output,
        params=params,
        bond_dim=args.bond_dim,
        projection_length=args.projection_length,
        fit_degree=args.fit_degree,
        results=results,
        pair_separations=args.pairs,
        triplets=args.triplets,
        max_residual_rms=args.max_residual_rms,
        max_loss=args.max_loss,
    )

    print("\nThermodynamic-limit iMPS linear-response summary")
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
