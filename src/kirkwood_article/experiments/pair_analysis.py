"""Compare adaptive pair-correlation simulations with iMPS linear response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kirkwood_article.analysis.density_bias import analyze_density_bias
from kirkwood_article.experiments.adaptive_d_scaling import (
    load_run_summaries,
    save_pcf_posthoc_analysis,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    return value


def validate_kernel_compatibility(
    simulation_summary: dict[str, Any], imps_summary: dict[str, Any], allow_mismatch: bool = False
) -> None:
    """Validate that simulation and iMPS outputs use compatible kernels."""

    sim_config = dict(simulation_summary.get("config", {}))
    sim_kernel = sim_config.get("kernel") or simulation_summary.get("kernel")
    imps_model = dict(imps_summary.get("model", {}))
    imps_kernel = imps_model.get("kernel_type", "exponential")
    if allow_mismatch:
        return
    if sim_kernel and sim_kernel != imps_kernel:
        raise ValueError(
            f"kernel mismatch: simulation uses {sim_kernel!r}, iMPS uses {imps_kernel!r}; "
            "pass --allow-kernel-mismatch for exploratory overlays"
        )


def _imps_pair_arrays(imps_summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pairs = imps_summary["pair_derivatives"]
    rows = sorted((float(item["distance"]), float(item["g2_prime"])) for item in pairs.values())
    if not rows:
        raise ValueError("iMPS summary contains no pair derivatives")
    return np.asarray([r[0] for r in rows], dtype=float), np.asarray([r[1] for r in rows], dtype=float)


def build_pair_analysis(
    simulation_dir: Path,
    imps_summary_path: Path,
    *,
    allow_kernel_mismatch: bool = False,
) -> dict[str, Any]:
    """Build derived iMPS prediction arrays aligned with simulation outputs."""

    aggregate_summary_path = simulation_dir / "summary.json"
    simulation_summary = _load_json(aggregate_summary_path) if aggregate_summary_path.exists() else {}
    imps_summary = _load_json(imps_summary_path)
    validate_kernel_compatibility(simulation_summary, imps_summary, allow_kernel_mismatch)

    summaries = load_run_summaries(simulation_dir)
    density_analysis = analyze_density_bias(summaries)
    d_values = np.asarray(density_analysis["d_values"], dtype=float)
    observed_density = np.asarray(density_analysis["observed"], dtype=float)
    density_ci_half_width = np.asarray(density_analysis["ci_half_width"], dtype=float)
    mean_field_density = np.asarray(density_analysis["expected"], dtype=float)

    pcf_path = simulation_dir / "pcf_posthoc_analysis.json"
    if not pcf_path.exists():
        save_pcf_posthoc_analysis(simulation_dir)
    pcf = _load_json(pcf_path)
    radii = np.asarray(pcf["radii"], dtype=float)
    pcf_excess_mean = np.asarray(pcf["pcf_excess_mean"], dtype=float)
    pcf_excess_lower = np.asarray(pcf["pcf_excess_lower"], dtype=float)
    pcf_excess_upper = np.asarray(pcf["pcf_excess_upper"], dtype=float)

    imps_model = imps_summary["model"]
    rho0 = float(imps_model["lattice_rho0"])
    rho_prime = float(imps_summary["rho_prime"])
    density_prediction = rho0 + rho_prime * d_values
    pair_distances, pair_g2_prime = _imps_pair_arrays(imps_summary)
    imps_g2_prime_on_radii = np.interp(radii, pair_distances, pair_g2_prime, left=np.nan, right=np.nan)
    imps_pcf_excess_prediction = d_values[:, None] * imps_g2_prime_on_radii[None, :]

    payload = {
        "d_values": d_values,
        "density": {
            "observed": observed_density,
            "ci_half_width": density_ci_half_width,
            "mean_field": mean_field_density,
            "imps_linear_response": density_prediction,
            "imps_rho_prime": rho_prime,
        },
        "pair": {
            "radii": radii,
            "simulation_pcf_excess_mean": pcf_excess_mean,
            "simulation_pcf_excess_lower": pcf_excess_lower,
            "simulation_pcf_excess_upper": pcf_excess_upper,
            "imps_pair_distances": pair_distances,
            "imps_g2_prime": pair_g2_prime,
            "imps_g2_prime_on_radii": imps_g2_prime_on_radii,
            "imps_pcf_excess_prediction": imps_pcf_excess_prediction,
        },
        "kernel": {
            "simulation": simulation_summary.get("config", {}),
            "imps": imps_model,
            "allow_kernel_mismatch": allow_kernel_mismatch,
        },
        "imps_diagnostics": imps_summary.get("method", {}),
    }
    return payload


def save_pair_analysis_outputs(
    simulation_dir: Path,
    imps_summary_path: Path,
    *,
    output_prefix: Path | None = None,
    allow_kernel_mismatch: bool = False,
) -> Path:
    """Save JSON/NPZ summaries and overlay plots for pair analysis."""

    payload = build_pair_analysis(
        simulation_dir, imps_summary_path, allow_kernel_mismatch=allow_kernel_mismatch
    )
    if output_prefix is None:
        output_prefix = simulation_dir / "pair_analysis"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    d_values = np.asarray(payload["d_values"], dtype=float)
    density = payload["density"]
    pair = payload["pair"]
    radii = np.asarray(pair["radii"], dtype=float)

    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        d_values=d_values,
        density_observed=np.asarray(density["observed"], dtype=float),
        density_ci_half_width=np.asarray(density["ci_half_width"], dtype=float),
        density_mean_field=np.asarray(density["mean_field"], dtype=float),
        density_imps_linear_response=np.asarray(density["imps_linear_response"], dtype=float),
        radii=radii,
        simulation_pcf_excess_mean=np.asarray(pair["simulation_pcf_excess_mean"], dtype=float),
        simulation_pcf_excess_lower=np.asarray(pair["simulation_pcf_excess_lower"], dtype=float),
        simulation_pcf_excess_upper=np.asarray(pair["simulation_pcf_excess_upper"], dtype=float),
        imps_pair_distances=np.asarray(pair["imps_pair_distances"], dtype=float),
        imps_g2_prime=np.asarray(pair["imps_g2_prime"], dtype=float),
        imps_pcf_excess_prediction=np.asarray(pair["imps_pcf_excess_prediction"], dtype=float),
    )
    json_path = output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(payload), fh, indent=2, sort_keys=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].errorbar(
        d_values,
        density["observed"],
        yerr=density["ci_half_width"],
        fmt="o",
        capsize=3,
        color="tab:blue",
        label="simulation",
    )
    axes[0].plot(d_values, density["mean_field"], "--", color="black", label="mean field")
    axes[0].plot(
        d_values,
        density["imps_linear_response"],
        color="tab:green",
        label="iMPS linear response",
    )
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    positive = d_values > 0
    if np.any(positive):
        sim_slope = np.nanmean(
            np.asarray(pair["simulation_pcf_excess_mean"], dtype=float)[positive]
            / d_values[positive, None],
            axis=0,
        )
        axes[1].plot(radii, sim_slope, color="tab:blue", label="simulation slope")
    axes[1].plot(
        pair["imps_pair_distances"],
        pair["imps_g2_prime"],
        color="tab:green",
        label="iMPS prediction",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("distance x")
    axes[1].set_ylabel("d[g(x)-1]/dd at d=0")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_summary.png"), dpi=200)
    plt.close(fig)

    return json_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path, required=True)
    parser.add_argument("--imps-summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--allow-kernel-mismatch", action="store_true")
    args = parser.parse_args(argv)
    save_pair_analysis_outputs(
        args.simulation_dir,
        args.imps_summary,
        output_prefix=args.output_prefix,
        allow_kernel_mismatch=args.allow_kernel_mismatch,
    )


if __name__ == "__main__":
    main()
