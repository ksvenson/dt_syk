# Production parameters, defaults, and cache policy

Run `python syk_temp_proj.py --help` to see every option and its default.
N and NA count Majorana modes. Times use hbar=1. The initial state is the
vacuum product state; this is fixed, not a command-line parameter.

| Option | Default | Meaning |
|---|---|---|
| `--N` | Required | Even total Majorana count, greater than four |
| `--g` | Required | Hamiltonian is H4+(g/N)H2 |
| `--NA` | 4 | Even subsystem Majorana count; DA=2^(NA/2) |
| `--parity` | `even` | Diagonalize even parity; `full` is available for checks |
| `--k-max` | 3 | Compute moments 1 through k_max |
| `--realizations` | 128 | Independent disorder realizations, regardless of MPI rank count |
| `--tmax` | 100000 | Last reporting time |
| `--dt` | `auto` | Maximum temporal-overlap integration spacing; accepts a positive number |
| `--check-dt` | Off | Repeat temporal integration with half the requested maximum step |
| `--projected-sampling` | `stratified` | One uniform time per interval; `grid` uses endpoints |
| `--projected-dt` | 1 | Maximum width of projected sampling/reporting intervals |
| `--chunk` | 32 | Number of projected times processed together; controls temporary memory |
| `--n-scr` | 1000 | Haar input kets for higher-order Scrooge targets; first order is exact |
| `--seed` | 247673152 = 628^3 | Base seed for disorder, Haar inputs and time sampling |
| `--time-seed` | 0 | Varies projected time draws without changing disorder or Haar inputs |
| `--cache-dir` | `./computation_cache/` | Explicit directory for intermediate caches, result archives and diagnostics |
| `--cache-hamiltonian` / `--no-cache-hamiltonian` | Off | Enable/disable dense Hamiltonian cache reads and writes |
| `--cache-eigensystem` / `--no-cache-eigensystem` | On | Enable/disable eigenvalue/eigenvector caching |
| `--cache-projected` / `--no-cache-projected` | Off | Enable/disable redundant per-state intermediate caches |
| `--degeneracy-tolerance` | Automatic machine-precision scale | Absolute energy-block tolerance for diagnostics and coherent energy dephasing |
| `--gap-k-max` | 1 | Compute minimum k-sum gaps through this order, up to k_max; 0 disables |
| `--gap-max-sums` | 5000000 | Skip an order if its number of k-sums exceeds this cap |
| `--gap-lines` | Off | Plot resolved, in-window median inverse-gap references on temporal curves |
| `--thermalization-fraction` | None | Mark and save observed crossings below a chosen fraction of each initial disorder-mean distance |
| `--verbose` | Off | Print per-realization details from all ranks; otherwise only rank-zero run summary and warnings |
| `--no-plots` | Off | Save results/diagnostics without rendering figures |
| `--help` | — | Print usage, descriptions and defaults |

## Projected sampling

With `projected_dt=1` and an integer tmax, stratified mode draws one independent
uniform time from [0,1), another from [1,2), etc. The state at time zero is
calculated separately. At reporting time j, the mean of the first j sampled
distances estimates the continuous uniform time average over [0,j]. It does
not include the zero-time distance in this mean. Randomized times avoid locking
the sample to a fixed lattice, in expectation; sampling noise remains.

Grid mode evaluates 0,1,2,... and includes zero in each cumulative discrete
mean. It is the earlier discrete projected-time averaging convention.
Neither mode changes the continuous temporal/RP integration.

The actual interval width is tmax/ceil(tmax/projected_dt), at most projected_dt,
so tmax is an exact endpoint. Halving projected_dt approximately doubles the
number of full-state projection calculations. Halving dt affects only scalar
spectral overlap integration. dt is not an ODE solver timestep.

## Cache policy and cluster behavior

`utilities.CACHE_DIR` remains a static default. Each decorated call accepts an
explicit `cache_dir` and optional `cache_enabled`; the decorator consumes these
arguments without altering the numerical function. Supplying a directory does
not reassign module configuration. Files are written atomically within that
directory. Separate runs should still use distinct notes for distinct physical
inputs; there is no automatic comparison of input matrices.

The production driver checks for a saved eigensystem **before** constructing
or loading H. On a cache hit, H is not needed at all. On a miss, H is constructed
and diagonalized, and by default only the eigensystem is saved. Scrooge targets
are small and remain cached; per-state projected intermediates are optional.
Combined result archives and per-realization diagnostics are always saved.

At N=24 in even parity, a complex128 dense H occupies approximately 64 MiB.
128 such files occupy approximately 8 GiB, excluding filesystem overhead.
Eigenvectors occupy roughly another 64 MiB per realization and are the useful
restart data. Removing the redundant H files therefore saves about half the
matrix-cache storage while retaining repeat-analysis speed. Full-space matrices
are four times as large. Disabling eigensystem caching saves more storage but
requires diagonalization again whenever analysis is repeated.

The production default does not read existing H caches when H caching is off.
The older Hamiltonian factory functions still support caching for older scripts;
pass `cache_enabled=False` to bypass them explicitly. This change does not
remove any existing files. A cache cleanup remains the user's separate action.

`functools.lru_cache` is separate from this on-disk cache. It retains only the
symmetric-basis data within each running Python process. There is no shared
file, cluster communication or persistence between jobs. Each MPI process keeps
its own copy. The basis cache is bounded to 32 parameter combinations and the
full-basis conversion cache to eight. The returned arrays are read-only so a
caller cannot accidentally change a basis used by later calculations. Large
DA/k can still make an individual conversion matrix large; bounding entry count
does not bound the number of bytes per entry.

## Timescale references and comparisons

To add spectral references through second order and a relative-distance marker:

```bash
python syk_temp_proj.py --N 24 --g 10 --gap-k-max 2 --gap-lines \
  --thermalization-fraction 0.1
```

The spectral diagnostic uses unordered k-tuples **with repeated indices allowed**,
matching the temporal moment. Different permutations of one tuple are trivial;
distinct tuples with equal sums yield zero gap. Only exactly unoccupied levels
are omitted. It sorts M=binomial(D_occupied+k-1,k) sums and checks neighboring
differences, rather than comparing all M squared pairs. At occupied dimension
2048, second order has 2098176 sums, while third order has 1433753600 sums.
The default cap permits the former and skips the latter before allocating it.
All calculations also report a floating-point resolution scale. Zero or
unresolved gaps are recorded without a finite inverse time; they are not
replaced with the smallest positive gap.

A resolved gap Delta gives a reference t_gap=2*pi/Delta: the first zero of
that individual sinc-squared contribution to temporal purity. The plots show
the median of finite, resolved reference times across disorder, only if it lies
inside the measured window. JSON files record skipped/unresolved orders and
NPZ arrays retain each realization's minimum gaps and reference times. The
median can exclude unresolved realizations, so consult those diagnostics.
These markers are not applied to projected curves, whose normalization changes
the frequency structure.

The minimum gap does not alone set the thermalization time. Its contribution
also depends on the energy populations and multiplicity weights. A tiny gap
with a tiny weight can be invisible at the desired accuracy. Without resonances,
the exact temporal formula gives the conservative bound

    d_T,k(tau) <= 2*sqrt(1-P_RP,k)/(tau*Delta_min,k).

This establishes the relevance of inverse gaps as a slowest-frequency bound,
but can substantially overestimate the time required for the actual distance
curve to fall below a chosen tolerance. The smallest gap and this bound are
sensitive to the occupied spectrum and numerical precision.

For direct curve comparison, `thermalization_fraction=f` reports the first
positive endpoint where the **disorder-mean distance** is <=f times its initial
value and stays below that level at all remaining sampled endpoints. It is an
observed finite-window criterion; recurrences after tmax are not excluded.
Absent crossings are stored as NaN, not assigned tmax. Temporal and projected
criteria are evaluated separately. A nonzero projected plateau or finite Haar
sampling error can prevent a requested threshold from being reached. Refine
time sampling and Scrooge samples before interpreting that as a physical failure.
Threshold times are saved even with `--no-plots`.

Per-realization diagnostics also record the populated energy standard deviation
sigma_E and widths 1/sqrt(k*sigma_E^2). They follow from
|A(u)|^(2k)=1-k*sigma_E^2*u^2+... and describe initial overlap decay, not convergence
of the kth temporal or projected ensemble to its target.

Saved-archive plotting supports the same optional annotations:

```bash
python plot_N_g_sweep.py --results tacc_cache/results*.npz \
  --gap-lines --thermalization-fraction 0.1
```

Archived annotation settings are used unless a fraction is overridden. Archives
without spectral diagnostics still plot normally. See WORKFLOW.md for the
quantities represented by the two curves.


## Console output and saved diagnostics

Default output consists of a rank-zero startup line, completion time, result
archive location, and the per-realization diagnostic-file pattern. If any
realization exceeds the conservative temporal sampling bound, rank zero prints
one combined warning. No per-rank gap dictionaries or refinement arrays are
printed by default. `--verbose` restores those details from each rank.

Every completed realization writes `timing_<run parameters>_r<index>.json`,
containing stage durations, eigensystem-cache status, run parameters, bandwidth,
sampling bound, effective integration step, energy spread, short-time overlap
widths, gap values/statuses, warnings, and refinement errors when requested.
Verbosity never controls whether these files are written. The result NPZ also
records warning flags per realization and analysis wall time, in addition to
curves, reporting axes, gap references, threshold times and parameters. Analysis
wall time excludes final archive writing and plotting; the Slurm timing CSV
continues to record the complete launcher duration, including plotting.

See NORM_DIAGNOSTICS.md for trace versus Frobenius norm bounds, degeneracy
floor estimates, and preservation of coherences in the energy-dephased target.
