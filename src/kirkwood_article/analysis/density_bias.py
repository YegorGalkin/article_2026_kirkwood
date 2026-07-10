"""Density-bias post-processing for adaptive d-scaling summaries."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ci_half_width_to_se(half_width: np.ndarray, batch_count: np.ndarray) -> np.ndarray:
    """Convert 95% CI half-widths back to standard errors."""

    critical_values = stats.t.ppf(0.975, np.maximum(batch_count - 1, 1))
    return np.divide(
        half_width,
        critical_values,
        out=np.full_like(half_width, np.nan, dtype=float),
        where=critical_values > 0,
    )


def weighted_zero_bias_fit(
    d_values: np.ndarray, residuals: np.ndarray, standard_errors: np.ndarray, quadratic: bool
) -> dict[str, float | list[float]]:
    """Fit residuals against ``d`` with intercept constrained to zero."""

    columns = [d_values]
    if quadratic:
        columns.append(d_values**2)
    design = np.column_stack(columns)
    weights = np.divide(
        1.0,
        standard_errors**2,
        out=np.ones_like(standard_errors, dtype=float),
        where=standard_errors > 0,
    )
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_residuals = residuals * np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_residuals, rcond=None)
    fitted = design @ coefficients
    weighted_sse = float(np.sum(weights * (residuals - fitted) ** 2))
    degrees_of_freedom = len(residuals) - len(coefficients)
    if degrees_of_freedom > 0:
        covariance = np.linalg.pinv(weighted_design.T @ weighted_design) * (
            weighted_sse / degrees_of_freedom
        )
        coefficient_se = np.sqrt(np.diag(covariance))
        t_values = np.divide(
            coefficients,
            coefficient_se,
            out=np.full_like(coefficients, np.nan, dtype=float),
            where=coefficient_se > 0,
        )
        p_values = 2.0 * stats.t.sf(np.abs(t_values), degrees_of_freedom)
    else:
        coefficient_se = np.full_like(coefficients, np.nan, dtype=float)
        p_values = np.full_like(coefficients, np.nan, dtype=float)
    result: dict[str, float | list[float]] = {
        "coefficients": [float(value) for value in coefficients],
        "standard_errors": [float(value) for value in coefficient_se],
        "p_values": [float(value) for value in p_values],
        "degrees_of_freedom": float(degrees_of_freedom),
        "weighted_sse": weighted_sse,
    }
    if quadratic:
        result["quadratic_coefficient"] = float(coefficients[1])
        result["quadratic_p_value"] = float(p_values[1])
        result["quadratic_significant_95_percent"] = bool(
            np.isfinite(p_values[1]) and p_values[1] < 0.05
        )
    return result


def analyze_density_bias(
    summaries: list[dict[str, float | int | bool]],
) -> dict[str, np.ndarray | dict[str, float | list[float]]]:
    """Analyze mean-density residuals relative to the zero-bias mean-field prediction."""

    ordered = sorted(summaries, key=lambda item: float(item["d"]))
    d_values = np.asarray([float(item["d"]) for item in ordered], dtype=float)
    observed = np.asarray([float(item["density_mean"]) for item in ordered], dtype=float)
    expected = np.asarray([float(item["mean_field_density"]) for item in ordered], dtype=float)
    ci_half_width = np.asarray([float(item["density_ci_half_width"]) for item in ordered], dtype=float)
    batch_count = np.asarray([int(item["batch_count"]) for item in ordered], dtype=float)
    standard_errors = ci_half_width_to_se(ci_half_width, batch_count)
    residuals = observed - expected
    linear = weighted_zero_bias_fit(d_values, residuals, standard_errors, quadratic=False)
    quadratic = weighted_zero_bias_fit(d_values, residuals, standard_errors, quadratic=True)
    return {
        "d_values": d_values,
        "observed": observed,
        "expected": expected,
        "ci_half_width": ci_half_width,
        "standard_errors": standard_errors,
        "residuals": residuals,
        "linear": linear,
        "quadratic": quadratic,
    }
