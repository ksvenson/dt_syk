# Numerical and performance review

## Is dt=1 too large?

Evolution is spectral: exp(-i E t) is evaluated directly. There is no time
integration error in the evolved state. The original `dt` controls a *regular
sampling lattice*, and its temporal moment is that of a discrete ensemble of
states. It is not automatically a converged approximation to the continuous
uniform temporal ensemble.

Write A(u) = sum_alpha p_alpha exp(-i E_alpha u). The highest possible frequency
of |A(u)|^(2k) is k W, where W = E_max - E_min in the occupied spectrum. A
conservative sufficient sampling bound is dt < pi/(k W). This does not prove a
particular plotted anomaly is aliasing: small weights, nearby folded frequency
resonances, finite averaging time, and Scrooge sampling noise also matter. The
code uses the entire computed sector spectrum for a safe bound, without
thresholding small populations. Projection involves normalized states, so this
finite frequency bound does **not** apply to projected distances.

For N=24, J=1, realization 0, base seed 628^3, even parity, an independent NumPy
reconstruction of the repository's Majorana and coupling convention gives:

| g | W | k=3 sufficient bound |
|---:|---:|---:|
| 0 | 12.946966 | dt < 0.080884 |
| 1 | 12.982183 | dt < 0.080664 |
| 10 | 15.259279 | dt < 0.068627 |
| 100 | 78.355908 | dt < 0.013365 |

Thus dt=1 substantially exceeds the conservative resolution bound, and even
0.1 exceeds it for these examples. `benchmark_kernels.py --bandwidth-N 24`
reproduces this table. The NumPy construction was checked against explicit dense
Pauli products at N=8. A separate test checks it against dynamite when that
backend is installed. These numbers are one realization, not a sweep-wide bound.

## Sampling without a tenfold projection cost

The production script now separates three quantities:

* `--dt`: maximum quadrature spacing for *spectral overlaps only*, O(D_sector)
  per sample. Default `auto` uses min(0.1, pi/(2 k_max W)). The actual spacing is
  rounded down to obtain an even number of subdivisions per reporting interval.
* `--projected-dt`: width of strata used to sample full states and projected
  distances; defaults to 1, so decreasing temporal dt does not add full states.
* Reporting times: stratum endpoints from 0 through `--tmax`, saved in the result.

Continuous temporal purity is evaluated through

    tr[rho_T^(k)(tau)^2] = 2/tau^2 integral_0^tau (tau-u)|A(u)|^(2k) du.

The squared temporal/RP Frobenius distance is this purity minus the exact RP
purity. Simpson quadrature streams overlaps and cumulative integrals in bounded
memory. No exponentially large moment matrix or k-sum-pair array is needed.
`--check-dt` recomputes at half the requested spacing and records the maximum
absolute changes, in both squared distance and distance, per moment order. The
Nyquist bound is a resolution criterion, not a requested quadrature tolerance;
verify refinement over the times relevant to the plots.

For projection, the default is one uniformly random time per stratum, with an
independent reproducible RNG stream. Each positive-time prefix estimates the
continuous average of the **distance for a state drawn from the temporal
ensemble**, as in the original workflow. It does not instead take the distance
of a time-averaged projected moment. Random sampling removes fixed-lattice bias
in expectation; the tradeoff is statistical noise. Initial time is handled
separately, and each prefix averages only the strata covering [0,tau]. Vary
`--time-seed` to assess temporal sampling uncertainty and compare
`--projected-dt 1` with `--projected-dt 0.5` on a pilot. `--projected-sampling grid`
retains the original discrete projected-time averaging convention for comparison;
the temporal curve remains a continuous quadrature estimate in either mode.

## Implemented speedups and fixes

1. Even-parity Hamiltonian and eigensystem by default: at N=24 the dense dimension
   is 2048 instead of 4096. Matrix storage is quartered; dense diagonalization
   has roughly one eighth of the cubic work. This is valid for the current |0>
   initial state and even-body Hamiltonian. Arbitrary states spanning both
   sectors require different handling.
2. Explicit sector-to-computational-basis mapping. A compact parity vector cannot
   be reshaped as bath x subsystem. Only evolved batches are expanded before
   projection; eigenvectors remain compact.
3. Orthonormal symmetric moment coordinates of size binomial(DA+k-1,k), rather
   than DA^k. For DA=4,k=3 this is 20 instead of 64, with identical Frobenius
   distances. Weighted tensor features turn the moment contraction into batched
   matrix products and remove the expensive multi-input einsum.
4. BLAS matrix multiplication reconstructs evolved states instead of the original
   unoptimized two-input einsum.
5. Conditional sigma_{A|z} blocks are formed directly. The full D x D diagonal
   ensemble density matrix is no longer constructed using cubic contraction.
6. Scrooge sampling uses normalized complex Gaussian Haar vectors instead of
   full Haar unitary matrices; bath blocks are processed in bounded batches.
   Its k=1 moment is calculated analytically. Higher-order sampling preserves
   the probability-adjusted Haar measure and the original shared Haar sample
   across bath outcomes, though individual Monte Carlo samples change.
7. Zero-probability projected outcomes contribute zero rather than NaN from
   division by zero. This matters at t=0 and in parity-constrained states.
8. Realization count is independent of rank count; gathered results are restored
   to realization order. This enables fair comparisons between MPI allocations.
9. Per-realization stage timings and sampling diagnostics are saved as JSON;
   the combined NPZ contains time axes, moment orders, parameters, and both
   distances. Cache names distinguish parity and sampling. Hamiltonian caching is opt-in;
   eigensystem hits skip Hamiltonian construction entirely. Cache cleanup is
   required before reusing names from a different coupling convention. Timing includes cache reads if
   caches already exist; use a new `--cache-dir` for cold-run comparisons.
10. The optional legacy energy-gap histogram sorts k-subset sums and counts
    pairs with binary searches instead of enumerating all pairs. It handles
    floating-point histogram edges, uses int64 counts, returns zero for distinct
    equal sums, and fixes the erroneous MPI minimum reduction. Its subsets still
    exclude repeated indices, as before; the exact temporal formula uses multisets.
    The latter now constructs multisets directly rather than sorting D^k tuples.
    Both enumeration paths reject excessive sum counts before allocation.
11. Odd N previously silently became N-1 via integer division. It is rejected;
    the default sweep now uses 16,18,20 instead of 16,17,19. The Slurm wrapper
    forwards analysis flags; the old plotting path now fills both missing arrays
    with NaN, and the new archive plotting path uses saved metadata/time axes.

## Is 5.5 hours reasonable?

It is plausible for the original implementation, but there are major avoidable
costs. Its N=24 state reconstruction performs approximately 100,000 x 4096^2
complex multiply-adds per realization. The k=3 moment contraction processes
100,000 x 1024 bath outcomes x 4^6 tensor entries, with repeated symmetric
entries. The Slurm script launches 128 single-threaded MPI ranks; the original
Python script assigns one independent disorder realization to each rank. This
is an ensemble of 128 realizations sharing the node, not one realization using
128 cores for diagonalization. Memory and bandwidth contention can be substantial.

Local single-thread measurements (minimum of three runs, synthetic arrays,
NumPy 2.3.5/SciPy 1.17.0) verified outputs and gave approximately:

| Kernel | Workload | Original | New | Ratio |
|---|---|---:|---:|---:|
| Projected k=3 moment | 8 times, DB=1024, DA=4 | 0.471 s | 0.00326 s | 145x |
| Conditional covariance | D=512, DA=4 | 0.135 s | 0.00141 s | 96x |
| State reconstruction | 32 times, D=512 | 0.00765 s | 0.00156 s | 4.9x |

These are kernel ratios on this environment, **not** an end-to-end or cluster
speedup. The new finer overlap grid adds work, and the relative bottlenecks
change. The 48-hour budget cannot be certified without a representative cluster
pilot. Keep one BLAS thread per rank with the current allocation; do not increase
BLAS thread counts without reducing MPI ranks and respecting available memory.

## Validation and suggested cluster pilot

Local checks: independent full-tensor versus symmetric moments and distances;
full versus parity-mapped state evolution/projection; direct covariance blocks;
Scrooge quadrature formula with identical Haar samples; analytic k=1;
random-phase recurrence versus brute-force moments; Simpson temporal quadrature
versus exact energy-sum sinc formulas through k=3, including resonances;
histograms versus explicit pair enumeration; and reference Hamiltonians versus
dense Pauli products. Eighteen tests pass, including cache scheduling and timescale diagnostics. One dynamite factory test is
skipped here because dynamite/PETSc/SLEPc are unavailable. Python compilation
and shell syntax checks pass. A serial orchestration smoke test using NumPy
Hamiltonian and communicator substitutes also produced matching full/even
projected curves and valid saved-archive plots. This does not validate real MPI.

In the configured cluster environment, first run the backend test:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Then run a short N=24 pilot using the existing launcher, reserving the intended
node/rank count. Keep all parameters fixed between comparisons. For example,
under a 128-rank allocation:

```bash
ibrun python3 syk_temp_proj.py --N 24 --g 10 --realizations 128 \
  --tmax 1000 --dt auto --projected-dt 1 --check-dt \
  --cache-dir computation_cache/pilot-g10 --no-plots
```

Inspect per-rank JSON stage timings, dt refinement errors, and peak memory from
Slurm accounting. Projection and temporal sampling approximately scale with
T; Hamiltonian construction, diagonalization and Scrooge target construction
are one-time costs. Include plotting/I/O and slowest-rank margins when estimating
a T=100,000 production job. Validate a smaller case with `--parity full` using
separate caches, and test the projected sampling seed/width before interpreting
fine features. Repeat the pilot at the largest g, because its overlap step can
be substantially smaller. Scrooge sampling uncertainty is not included in the
plotted disorder SEM; check n_scr refinement independently.

The older `projected_computation.py` refers to removed/nonexistent `proj_moment`
and `pf.norm` APIs and is not the production entry point. The older exact
moment/histogram scripts remain combinatorially expensive at large D, even with
their improved setup/counting. Use the production overlap-based path for N=24.
