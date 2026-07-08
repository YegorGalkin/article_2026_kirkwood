"""Figure generation for scaling experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_scaling(results_path: Path, output: Path) -> None:
    """Create a two-panel density and pair-correlation comparison figure."""

    data = np.load(results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(
        data["d_values"], data["density_mean"], yerr=data["density_se"], fmt="o", label="SSA"
    )
    axes[0].plot(data["d_values"], data["theory_density"], label="mean-field theory")
    axes[0].set_xlabel("additional death rate d")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].errorbar(
        data["radii"], data["pcf_mean"][0], yerr=data["pcf_se"][0], fmt="o", label="SSA"
    )
    axes[1].plot(data["radii"], data["theory_pcf"], label="independent baseline")
    axes[1].set_xlabel("distance r")
    axes[1].set_ylabel("g(r)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("article/figures/scaling.png"))
    args = parser.parse_args(argv)
    plot_scaling(args.results, args.output)


if __name__ == "__main__":
    main()
