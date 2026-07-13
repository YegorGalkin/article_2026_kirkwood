#!/usr/bin/env python3
"""Basic consistency tests for imps_linear_response.py."""

import importlib.util
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "imps_linear_response", HERE / "imps_linear_response.py"
)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    params = MOD.ModelParameters(b=1.0, gamma=1.0, lam=1.0, a=0.25)
    data = MOD.enumerate_cylinder(ell=6, params=params, device=torch.device("cpu"), dtype=torch.float64)
    initial = MOD.product_initial_tensors(
        bond_dim=3,
        p1=params.p0,
        hidden_mixing=0.25,
        perturbation=1e-3,
        seed=1,
    )
    model = MOD.StochasticInfiniteMPS(initial, torch.float64, torch.device("cpu"))

    residual, diag = MOD.projected_stationarity_residual(model, data, params, d=0.0)
    A = diag["A"].detach().numpy()
    E = A[0] + A[1]
    pi = diag["pi"].detach().numpy()

    assert np.max(np.abs(E @ np.ones(E.shape[0]) - 1.0)) < 1e-12
    assert np.max(np.abs(pi @ E - pi)) < 1e-12
    assert abs(pi.sum() - 1.0) < 1e-12
    assert abs(float(diag["probs"].sum()) - 1.0) < 1e-12
    assert abs(float(diag["rho"]) - params.rho0_lattice) < 1e-12
    assert float(residual.abs().max()) < 1e-12

    obs = MOD.infinite_mps_observables(
        A=A,
        pi=pi,
        params=params,
        pair_separations=[1, 2, 5],
        triplets=[(1, 2), (2, 5)],
    )
    assert max(abs(x - 1.0) for x in obs["g2"].values()) < 1e-12
    assert max(abs(x) for x in obs["k3_connected"].values()) < 1e-12

    print("All infinite-iMPS consistency tests passed.")


if __name__ == "__main__":
    main()
