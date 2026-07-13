# Thermodynamic-limit iMPS solver for DLBP linear response

This is the infinite-system replacement for the earlier finite-ring collocation
prototype.

The implementation follows the main structural ideas of MPSKit's
`InfiniteMPS`/VUMPS workflow:

- a one-site translation-invariant tensor is repeated on the infinite chain;
- the tensor is maintained in a canonical gauge;
- infinite left and right environments are solved directly;
- local projected fixed-point equations are imposed in the thermodynamic limit;
- density and correlation functions are evaluated from transfer-matrix
  contractions, with no finite-chain extrapolation.

Reference documentation:

- https://quantumkithub.github.io/MPSKit.jl/dev/#Infinite-Matrix-Product-States
- V. Zauner-Stauber et al., *Variational optimization algorithms for uniform
  matrix product states*, Phys. Rev. B 97, 045145 (2018).

## Why this is not a literal quantum VUMPS implementation

The DLBP stationary state is a probability vector of a non-Hermitian Markov
generator. Its exact left eigenvector is the flat summation state. Standard
quantum VUMPS instead optimizes a Hermitian Rayleigh quotient using the
Hilbert-space norm.

The script therefore uses the corresponding **classical stochastic iMPS**
canonical form. For physical state `s in {0,1}`, the matrices satisfy

```text
A[s] >= 0,
E = A[0] + A[1],
E @ 1 = 1.
```

The normalized left Perron vector `pi` is obtained from

```text
pi @ E = pi,
sum(pi) = 1.
```

Consequently, the probability of an infinite-chain cylinder word is

```text
P(x_0 ... x_{ell-1}) = pi A[x_0] ... A[x_{ell-1}] 1.
```

This is the classical analogue of a one-site right-canonical `InfiniteMPS`.

## Exact infinite exterior contraction

For the lattice exponential kernel

```text
K_m = kappa q^m,
q = exp(-lambda dx),
kappa = lambda / 2,
```

the two exterior half-lines are summed analytically with

```text
R_q = (I - q E)^(-1).
```

For a cylinder word `x` and a site `i` inside it, the script computes exactly

```text
F_i(x) = E[1_{cylinder=x} H_i],
```

including:

1. particles inside the cylinder;
2. every site in the infinite left half-line;
3. every site in the infinite right half-line.

There is no periodic wraparound or exterior-field moment closure.

## Infinite-volume projected stationarity equation

For every binary word of length `ell`, the script evaluates its exact master-
equation flux under the iMPS law. Only events inside the cylinder change the
word, while their rates use the complete infinite field from the resolvent
contraction.

The optimization minimizes the weighted residual of all `2**ell` word
stationarity equations. This is a thermodynamic-limit Galerkin/tangent-space
projection analogous in purpose to the local projected equations used by
VUMPS.

The systematic controls are:

- bond dimension `D`;
- cylinder projection length `ell`;
- lattice spacing `dx`;
- the small positive `d` values used for the derivative fit.

The cylinder length is a projection order, not a finite system size. The two
exterior regions remain infinite for every `ell`.

## Files

- `imps_linear_response.py`: new thermodynamic-limit iMPS solver.
- `imps_linear_response_finite_ring_legacy.py`: previous periodic-ring
  collocation implementation.
- `test_imps_infinite.py`: checks the exact PPP fixed point and canonical
  normalization.

## Typical run

From the repository root, prefer the packaged `uv` console entry point. The
repository uv configuration includes the `imps` dependency group by default, so
PyTorch is present for the iMPS entry point without extra flags:

```bash
uv sync
uv run run-imps-linear-response \
  --b 1 --gamma 1 --lam 1 \
  --a 0.25 \
  --bond-dim 3 \
  --projection-length 7 \
  --d-values 0.00125,0.0025,0.005,0.01 \
  --pairs 1,2,3,4,5,6,8,10 \
  --triplets '1,2;1,3;2,4;2,6' \
  --fit-degree 2 \
  --output data/imps_infinite/run_infinite_D3_l7
```

For a quick smoke test through the same entry point:

```bash
uv run run-imps-linear-response \
  --bond-dim 2 \
  --projection-length 4 \
  --d-values 0.005,0.01 \
  --pairs 1,2,3 \
  --triplets '1,2;1,3' \
  --adam-steps 20 \
  --lbfgs-steps 0 \
  --fit-degree 1 \
  --output data/imps_infinite/smoke_test
```

The legacy script path remains available as a wrapper for ad-hoc use:

```bash
uv run python scripts/iMPS/imps_linear_response.py --help
```

For comparison against adaptive simulations, generate the iMPS summary and then
use the main repository pair-analysis entry point:

```bash
uv run run-pair-analysis \
  --simulation-dir data/adaptive_d_scaling_exp_var1 \
  --imps-summary data/imps_infinite/run_infinite_D3_l7/summary.json
```

## Calculated responses

The density derivative is obtained from a fixed-intercept fit

```text
rho(d) = rho(0) + rho_prime d + O(d^2).
```

For pair separation `r=m dx`, the script returns

```text
d/d d k_2(r) |_{d=0},
d/d d g_2(r) |_{d=0}.
```

For sites `0,m,n`, the connected factorial triplet is

```text
kappa_3(0,m,n)
  = k_3(0,m,n)
    - rho [k_2(m) + k_2(n) + k_2(n-m)]
    + 2 rho^3.
```

The script returns its derivative and the derivative of the normalized version
`kappa_3/rho^3`.

## Active-branch selection

The vacuum remains a formal stationary iMPS. Continuation starts at the exact
positive-density `d=0` Bernoulli state. A lower-density barrier prevents a
numerical jump to the vacuum; it is exactly zero when the solution remains
above `--branch-fraction` times the unperturbed density.

Check that the reported solution is independent of decreasing
`--branch-fraction` and that the barrier is inactive.

## Convergence protocol

A useful sequence is:

```text
D = 2, 3, 4, ...
ell = 5, 6, 7, ...
dx = 0.5, 0.25, 0.125, ...
```

At each level verify:

- the projected residual RMS decreases;
- `rho_prime`, `g2_prime`, and `kappa3_prime` stabilize;
- the subleading transfer eigenvalues stabilize;
- changing the small-`d` fitting window does not materially change the slope.

Because the local hard-core lattice has

```text
rho_0(dx) = b / (gamma + b dx),
```

the continuum PPP density `b/gamma` is recovered only as `dx -> 0`.
