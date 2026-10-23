# Deep thermalization in SYK

The production entry point is `syk_temp_proj.py`: diagonalize (1-g) SYK4 + g SYK2,
compare continuous temporal moments with exact random-phase moments, and compare
state-conditioned projected moments with conditional Scrooge moments.

After diagonalization, the energy basis is adapted to the initial ket: each
degenerate energy block has one occupied basis ket. Independent RP phases
therefore act on energy-projected initial components, so the first temporal
moment converges to the RP first moment even with energy degeneracies.

See [INSTANTANEOUS_SCROOGE.md](INSTANTANEOUS_SCROOGE.md) for the two new
instantaneous targets, exact identities, sampling, and computational savings.

See [NORM_DIAGNOSTICS.md](NORM_DIAGNOSTICS.md) for degeneracies, moment norms,
and the energy-dephased reference.

See [PARAMETERS.md](PARAMETERS.md) for every command-line option, sampling
behavior, cache policy, and plot timescale diagnostics.

See [WORKFLOW.md](WORKFLOW.md) for the physical quantities, formulas, and
independent checks at each stage.

See [README.txt](README.txt) for the existing dynamite/PETSc/SLEPc installation
instructions and [PERFORMANCE.md](PERFORMANCE.md) for the numerical changes,
benchmarks, sampling conventions, and cluster validation steps.

```bash
# Small pilot in the configured dynamite environment
python syk_temp_proj.py --N 12 --g 0.5 --realizations 1 --tmax 100 --check-dt

# Existing TACC wrapper; dt controls overlaps, projected-dt controls full states
sbatch -t 48:00:00 syk_job.slurm 24 10 --dt auto --projected-dt 1

# Tests and local microbenchmarks need only NumPy and SciPy
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python benchmark_kernels.py --bandwidth-N 24

# Plot saved sweep archives using their stored time axes
python plot_N_g_sweep.py --results tacc_cache/results*.npz
```

N counts Majoranas and must be even. The initial state is the vacuum product
state `|0>`. Its dynamics are exactly confined to even fermion parity, which is
the default; `--parity full` is available for validation. There are 128 disorder
realizations by default, independently of the MPI rank count. No jobs are
submitted by the Python code.
