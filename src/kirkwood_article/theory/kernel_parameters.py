"""Shared kernel-parameter conventions for simulation/theory comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExponentialKernelParameters:
    """Parameters for ``K(r) = (lambda / 2) exp(-lambda |r|)`` in 1D."""

    variance: float = 1.0

    @property
    def lam(self) -> float:
        """Return lambda for the requested variance, using Var = 2 / lambda**2."""

        if self.variance <= 0.0:
            raise ValueError("variance must be positive")
        return math.sqrt(2.0 / self.variance)

    @property
    def std(self) -> float:
        """Return the standard deviation used by the SSA kernel parameter."""

        return math.sqrt(self.variance)

    @property
    def normalization(self) -> str:
        return "K(r)=(lambda/2)*exp(-lambda*abs(r)); variance=2/lambda**2"


def exponential_kernel_parameters(variance: float = 1.0) -> ExponentialKernelParameters:
    """Return shared 1D exponential-kernel parameters for theory/simulation."""

    return ExponentialKernelParameters(variance=variance)
