# Periodic mixed-field Ising analysis

`mfim_temp_proj.py` computes the same four distance curves as the SYK driver:

| Saved key | Quantity |
|---|---|
| `temp_rp_2norm` | Continuous temporal moment versus adapted RP moment |
| `proj_scr_2norm` | Running mean distance of a projected moment from fixed generalized Scrooge G_k[sigma] |
| `proj_single_scr_2norm` | Running mean distance from single Scrooge S_k[rho_A(t)] |
| `proj_single_scr_instant_2norm` | That same per-state single-Scrooge distance, without averaging over earlier times |

There is no comparison with generalized Scrooge of the instantaneous full pure
state. The fourth curve simply reuses an existing distance before averaging.
`projected_sample_times` stores the actual evaluation times; with stratified
sampling these lie inside the reporting intervals rather than at their endpoints.

## Hamiltonian and units

Here L counts spin-1/2 sites, LA counts subsystem sites, D=2^L and DA=2^LA.
These are different from the Majorana counts N and NA in the SYK command line.
In units with hbar=1, the driver fixes

$$H=\sum_{i=0}^{L-1}\left[J Z_i Z_{(i+1)\bmod L}+h_x X_i+h_z Z_i\right],
\qquad (J,h_x,h_z)=(1,0.9045,0.8090).$$

Kim, Ikeda and Huse use these parameters and periodic boundaries as their
nonintegrable Ising example in Eq. (7), Sec. II B of
[Phys. Rev. E 90, 052105 (2014)](https://doi.org/10.1103/PhysRevE.90.052105).
The existing factory already had these default fields and signs. Its historical
open-boundary default is preserved; the new driver explicitly requests periodic
boundaries. `MFIM` now accepts `periodic` as an alias for Dynamite's
[`closed` boundary convention](https://dynamite.readthedocs.io/en/stable/dynamite.operators.html#dynamite.operators.index_sum).
`MFIM_numpy` converts that same operator to a dense matrix. For real fields the
matrix is real symmetric, avoiding unnecessary complex eigensolver work.
There is no coupling sweep or analog of the SYK slider g.

The initial ket is the computational vacuum |0...0>, matching the SYK driver,
and is recorded explicitly in result metadata. The full spin Hilbert space is
used: the X field flips spins and does not conserve the SYK Z-basis fermion
parity. Translation and reflection symmetries of the periodic chain remain;
chaotic couplings do not eliminate these symmetries. The existing degeneracy
adaptation handles energy eigenspaces while retaining initial coherences. No
claim is made that arbitrary initial states or all moment orders must converge
to every Scrooge target; the vacuum need not represent infinite temperature.

A uses the LA least significant computational-basis spins, with full ket ordering
|z>_B tensor |a>_A, exactly as in the SYK projection code. No symmetry-reduced row
index is mistaken for a subsystem label.

## Shared calculation and MPI allocation

`ensemble_analysis.py` supplies common command-line controls, basis preparation,
diagnostics, time selection, running averages, threshold detection, and plotting.
Both drivers use the same `temporal_funcs`, `random_phase` and `projected_funcs`
kernels. SYK still assigns independent disorder realizations to ranks. MFIM has
one deterministic Hamiltonian per size, so its ranks instead share time work:

1. Rank zero constructs and diagonalizes H once, or loads the raw eigensystem.
   The cache name includes L, fields, J, periodic boundaries and full space.
2. Rank zero adapts the eigenbasis to the coherent initial ket once and builds
   the fixed generalized Scrooge target using those occupied eigenvectors.
3. The occupied eigenvectors, energies, coefficients and target moments are
   broadcast. Real eigenvectors stay real when applicable. Each rank holds its
   own copy; the eigensolver itself remains a rank-zero dense calculation.
4. Ranks integrate disjoint contiguous reporting intervals. For
   f_k(u)=|<psi0|exp(-iHu)|psi0>|^(2k), they return interval integrals of f_k(u)
   and u f_k(u), using a common Simpson spacing. The integrals are collected
   chronologically before forming the global temporal moment distance:

$$P_{T,k}(T)=2\left[\frac{1}{T}\int_0^T f_k(u)du
-\frac{1}{T^2}\int_0^T u f_k(u)du\right].$$

5. Ranks also evolve disjoint contiguous sets of sampled states and calculate
   per-state projected/fixed-target and projected/single-target distances.
   The partition of sample points includes t=0; the partition of integration
   intervals does not double-count it. Empty rank assignments are supported.
6. Rank zero collects per-state distances in time order and THEN forms the
   running averages. It does not average separate per-rank averages. Raw
   single-Scrooge distances remain available as the fourth curve.

The fixed Scrooge target uses RNG stream (0,1), and time selection uses
(0,2,time_seed), matching the SYK convention. Instantaneous Scrooge samples use
SeedSequence(seed,spawn_key=(3,global_sample_index)). Thus changing the MPI rank
count or time batch size does not change which random samples belong to a time.
MFIM has no disorder: changing seed changes Haar/time sampling, not H. There is
no `--realizations` option. Archives retain a leading realization axis of size
one for compatibility; plots do not invent a disorder error bar.

## Running and scheduling

For a small cluster pilot:

```bash
ibrun python3 mfim_temp_proj.py --L 8 --LA 2 --tmax 100 --n-scr 100 --check-dt
```

The common defaults include tmax=100000, n_scr=1000, k_max=3, projected_dt=1,
stratified projected times, and automatic temporal quadrature spacing. Shared
controls are registered by the same `add_sampling_arguments` function as SYK.
MFIM defaults to LA=2 spins, equivalent to SYK's NA=4 Majoranas, giving DA=4.
Automatic dt is spectrum dependent, so matching the `auto` setting does not
require the numerical integration step to match between Hamiltonians. Shared
cache switches, `--gap-k-max`, `--gap-lines`, `--thermalization-fraction`,
`--no-plots`, and `--verbose` are supported. Results and timing JSON use an
MFIM_L... prefix, so they do not collide with SYK output. There is one timing
JSON per MFIM run, including rank-by-rank state/interval counts and durations.

```bash
./submit_mfim_sweep.sh --dry-run --L 8 10 12 -- --LA 2 --tmax 100 --n-scr 100
./submit_mfim_sweep.sh --L 8 10 12 -- --LA 2 --tmax 100000
./submit_mfim_sweep.sh --L 12 --tasks 32 --nodes 1 --time 24:00:00 -- --check-dt
./collect_timings.sh
./collect_timings.sh --manifest mfim_manifest.csv
python3 plot_mfim_sizes.py --results computation_cache/results_MFIM_L*.npz
```

The submitter launches one independent job per L, with no coupling column or
nested coupling loop. It writes `mfim_manifest.csv`. `mfim_job.slurm` uses the
same PHY26019/normal allocation, .venv activation, one BLAS thread per rank,
`ibrun`, output directories, and per-job timing policy as the SYK scripts.
Analysis flags after `--` are forwarded unchanged. Resource defaults are
requests, not measured wall-time estimates; override them for cluster pilots.
`resources_for_L` near the top of `submit_mfim_sweep.sh` is an explicit table
with one row per spin count, in the form `L) echo "nodes tasks HH:MM:SS" ;;`.
Each L can have its own walltime, nodes and MPI task count. Edit the individual
rows for your measured runtimes, or use `--time`, `--nodes` and `--tasks` to
override those columns for every L in a submission. Sizes absent from the
table require either a new row or all three explicit resource overrides.
No Slurm jobs are submitted by tests or by a dry run.

The collector displays the SYK N/g and MFIM L/model timing schemas separately
and checks both manifests through sacct. `--manifest` limits which submission
manifest is checked in Slurm accounting. Job ids remain column 4 in both
manifests, including for jobs killed before writing their own timing file.
Existing timing files and manifests are not removed.

Dense exact diagonalization is still the size limit. A real full eigenvector
matrix occupies at most 8*4^L bytes per rank (2 GiB at L=14), before workspaces,
targets and temporary evolved states. Complex vectors require twice that.
More ranks replicate those vectors; adding ranks is not always a memory win.
For L>14 the submitter requires explicit nodes/tasks/time estimates. A separate
spatial-symmetry basis would be a further optimization, not currently applied.

Tests compare serial, 3-rank and 8-rank simulations, including empty assignments,
fixed sample streams, warm caches, and both grid/stratified averaging. They
verify the wraparound bond against an independent spin-matrix construction and
exercise job scripts with fake sbatch/ibrun/sacct commands. The real
Dynamite/PETSc/MPI backend still needs the supplied cluster pilot.
