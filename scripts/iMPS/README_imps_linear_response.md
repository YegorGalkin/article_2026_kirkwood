# Uniform-iMPS linear response for the 1D DLBP process

This directory contains `imps_linear_response.py`, a research prototype for the
small-intrinsic-death response of the one-species spatial birth-death process
with

\[
K(x)=\frac{\lambda}{2}e^{-\lambda |x|}.
\]

The continuum is discretized into binary cells of width `a=dx`. The lattice
rates are

\[
0\to1\text{ at site }i:\quad b\,a\sum_{j\ne i}K((i-j)a)n_j,
\]

\[
1\to0\text{ at site }i:\quad d+\gamma\sum_{j\ne i}K((i-j)a)n_j.
\]

At `d=0`, the exact hard-core product measure has odds

\[
\frac{p_0}{1-p_0}=\frac{ba}{\gamma},
\qquad
p_0=\frac{ba/\gamma}{1+ba/\gamma}.
\]

As `a -> 0`, its density tends to `b/gamma`.

## Ansatz

The stationary active state is represented by a nonnegative uniform MPS
(hidden-Markov matrix product distribution)

\[
P(n_1,\ldots,n_N)\propto
\operatorname{tr}\big[T_{n_1}\cdots T_{n_N}\big],
\qquad n_i\in\{0,1\}.
\]

The same tensors `T0,T1` are used at every site. Stationarity is enforced by
minimizing the exact master-equation residual on an enumerated periodic training
window. Infinite-chain observables are then evaluated using the Perron left and
right fixed points of `T=T0+T1`.

A finite ring with `d>0` ultimately reaches the vacuum. The code therefore uses
the nonempty-sector/quasi-stationary regularization: it conditions the training
ring on being nonempty and reflects only the singleton-to-vacuum intrinsic-death
transition. This has exponentially small effect when the expected number of
particles on the training ring is large.

## Quantities returned

For lattice separation `m` and physical separation `r=m*a`, the code evaluates

\[
\rho=\frac{\langle n_0\rangle}{a},
\]

\[
k_2(r)=\frac{\langle n_0n_m\rangle}{a^2},
\qquad
g_2(r)=\frac{k_2(r)}{\rho^2}.
\]

For sites `0,m,n`, `0<m<n`,

\[
k_3(0,ma,na)=\frac{\langle n_0n_mn_n\rangle}{a^3},
\]

and the connected factorial triplet density is

\[
\kappa_3(0,r,s)=k_3(0,r,s)
-\rho\,[k_2(r)+k_2(s)+k_2(s-r)]
+2\rho^3.
\]

The script reports

- `rho_prime`: \(\partial_d\rho|_{d=0}\),
- `g2_prime`: \(\partial_d g_2(r)|_{d=0}\),
- `k2_prime`: \(\partial_d k_2(r)|_{d=0}\),
- `connected_k3_prime`: \(\partial_d\kappa_3(0,r,s)|_{d=0}\),
- `normalized_connected_k3_prime`: derivative of \(\kappa_3/\rho^3\).

The derivative is obtained by solving at several small positive `d` values and
fitting

\[
y(d)=y(0)+c_1d+c_2d^2+\cdots
\]

with the exact `d=0` intercept fixed.

## Example

```bash
python imps_linear_response.py \
  --b 1 --gamma 1 --lam 1 \
  --a 0.25 \
  --bond-dim 3 \
  --n-train 14 \
  --d-values 0.00125,0.0025,0.005,0.01 \
  --pairs 1,2,3,4,5,6,8,10 \
  --triplets '1,2;1,3;2,4;2,6' \
  --fit-degree 2 \
  --exact-benchmark \
  --output run_D3_N14_a025
```

The exact benchmark solves the active-sector finite-ring Poisson equation at
`d=0`. It is limited by the `2**N` state space, but is useful for checking the
MPS optimization.

## Output files

- `summary.json`: fitted derivatives and run metadata.
- `continuation.csv`: observables at each small positive `d`.
- `tensors_and_data.npz`: optimized uniform tensors.
- `density_fit.png`: density continuation and constrained fit.
- `pair_response.png`: fitted pair-correlation derivative.

## Required convergence checks

A reported derivative should be regarded as converged only after checking all
of the following.

1. **Bond dimension:** repeat with `D=2,3,4,...`.
2. **Training length:** increase `N` while keeping `N*a` several kernel decay
   lengths, preferably `N*a >= 8/lambda`.
3. **Cell width:** decrease `a`, while increasing `N` so the physical training
   length does not shrink.
4. **Fitting window:** repeat with smaller maximum `d` and compare linear and
   quadratic constrained fits.
5. **Stationarity residual:** verify that the final losses decrease with `D`
   and optimization effort.
6. **Exact finite-ring comparison:** for feasible `N`, compare against
   `--exact-benchmark`.

A practical continuum sequence is, for example,

```text
(a, N) = (0.5, 20), (0.25, 40), (0.125, 80)
```

but exact enumeration of all training configurations then becomes impossible.
The current implementation is intended for moderate `N`; extending it to
Monte-Carlo residual collocation is the next scalability step.

## Interpretation

The method does not truncate the factorial-moment hierarchy. It truncates the
spatial entanglement/hidden-state dimension of the full probability law. A
converged result with

\[
\rho'(0)<-1/\gamma,
\quad g_2'(r)>0\text{ at short range},
\quad \kappa_3'(0,r,s)<0
\]

corresponds to stronger-than-mean-field density loss, pair clustering, and
triplet isolation.
