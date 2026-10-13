"""SYK4 + (g/N) SYK2: continuous temporal and sampled projected ensembles."""
import argparse
import json
import os
from time import perf_counter

import numpy as np
import dynamite as dm
import dynamite.subspaces as ss
import dynamite.tools as tools

import linalg as la
import hamiltonian_factory as hf
import temporal_funcs as tf
import projected_funcs as pf
import utilities as ut
import thermalization as th


def validate_N(arg):
    """Read N as an even number of Majorana modes greater than four.

    The spin representation has N/2 two-level sites, so odd N is rejected
    rather than silently replaced by N-1. Used to validate the --N argument."""
    num = int(arg)
    if num <= 4 or num % 2:
        raise argparse.ArgumentTypeError('N must be an even number of Majoranas greater than 4')
    return num


def positive_float(arg):
    """Read a finite, strictly positive real-valued analysis parameter.

    Used for time spacings and the maximum evolution time. Zero, negative
    values, NaN, and infinity are rejected before the calculation starts."""
    value = float(arg)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return value


def positive_int(arg):
    """Read a strictly positive integer analysis parameter.

    Used for sample counts, moment orders, and numbers of times handled
    together. Invalid values are rejected before the calculation starts."""
    value = int(arg)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def parse_args():
    """Read the physical and sampling parameters for a production run.

    Returns an argparse.Namespace whose fields have the names of the command
    options, with hyphens replaced by underscores. --N and --g specify the
    Hamiltonian; --NA fixes DA=2**(NA/2); --k-max requests orders 1 through k_max.
    --parity even is valid for the current vacuum initial state and even-body
    Hamiltonian. --realizations counts independent disorder realizations and
    is independent of the number of parallel processes.

    --dt sets the maximum temporal integration step; 'auto' selects it from
    the energy spectrum after diagonalization. --projected-dt independently
    sets the widths of intervals used to sample projected states. By default
    one time is drawn uniformly within each interval. --time-seed changes only
    these time draws, whereas --seed also changes disorder and Scrooge samples.
    --check-dt repeats temporal integration at a finer step to report accuracy.
    --n-scr sets the number of Haar inputs for each conditional Scrooge target.

    Run with --help for all controls. N and NA count Majorana modes, not
    Hilbert-space dimensions. NA must be even and no larger than N."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--N', type=validate_N, required=True, help='Even number of Majorana modes')
    parser.add_argument('--g', type=float, required=True, help='Coefficient g in H4+(g/N)H2')
    parser.add_argument('--dt', default='auto', help='Temporal quadrature max step: auto or a positive number')
    parser.add_argument('--projected-dt', type=positive_float, default=1., help='Projected stratum width; independent of --dt')
    parser.add_argument('--projected-sampling', choices=('stratified', 'grid'), default='stratified', help='stratified: one uniform random time per interval; grid: regular endpoints including zero')
    parser.add_argument('--tmax', type=positive_float, default=1e5, help='Maximum evolution/reporting time, hbar=1')
    parser.add_argument('--parity', choices=('even', 'full'), default='even', help='Eigenvector basis; even is valid for the vacuum initial state')
    parser.add_argument('--realizations', type=positive_int, default=128, help='Independent of MPI rank count')
    parser.add_argument('--n-scr', type=positive_int, default=1000, help='Haar inputs for higher Scrooge moments; k=1 is exact')
    parser.add_argument('--k-max', type=positive_int, default=3, help='Calculate moment orders 1 through this value')
    parser.add_argument('--NA', type=positive_int, default=4, help='Even subsystem Majorana count; DA=2**(NA/2)')
    parser.add_argument('--chunk', type=positive_int, default=32, help='Projected time batch size')
    parser.add_argument('--seed', type=int, default=628**3, help='Base seed for disorder, Haar states and time sampling')
    parser.add_argument('--time-seed', type=int, default=0, help='Independent stratified sampling replicate')
    parser.add_argument('--check-dt', action='store_true', help='Also compute dt/2 and record convergence error')
    parser.add_argument('--cache-dir', default=ut.CACHE_DIR, help='Explicit intermediate/results directory; leaves utilities.CACHE_DIR unchanged')
    parser.add_argument('--no-plots', action='store_true', help='Save results and diagnostics without rendering figures')
    parser.add_argument('--verbose', action='store_true', help='Print per-realization diagnostics from all ranks; detailed data is always saved')
    parser.add_argument('--cache-hamiltonian', action=argparse.BooleanOptionalAction, default=False,
                        help='Read/write dense H files; unnecessary when an eigensystem is saved')
    parser.add_argument('--cache-eigensystem', action=argparse.BooleanOptionalAction, default=True,
                        help='Read/write eigenvalues and eigenvectors for repeat analysis')
    parser.add_argument('--cache-projected', action=argparse.BooleanOptionalAction, default=False,
                        help='Save per-state projected intermediate arrays in addition to final results')
    parser.add_argument('--gap-k-max', type=int, default=1,
                        help='Exact spectral gap diagnostics through this order; 0 disables; capped by gap-max-sums')
    parser.add_argument('--gap-max-sums', type=positive_int, default=5_000_000,
                        help='Maximum number of k-sums to enumerate for each gap diagnostic')
    parser.add_argument('--gap-lines', action='store_true',
                        help='Show in-window median 2*pi/minimum-gap references on temporal plots')
    parser.add_argument('--thermalization-fraction', type=positive_float, default=None,
                        help='Mark sustained finite-window crossings below this fraction of the initial mean distance (0<f<1)')
    args = parser.parse_args()
    if args.NA % 2 or args.NA > args.N:
        parser.error('NA must be even and no larger than N')
    if not np.isfinite(args.g) or args.seed < 0 or args.time_seed < 0:
        parser.error('g must be finite; seeds must be nonnegative')
    if args.dt != 'auto':
        try:
            args.dt = positive_float(args.dt)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error('--dt must be auto or a finite positive number')
    if args.gap_k_max < 0:
        parser.error('--gap-k-max must be nonnegative')
    if args.thermalization_fraction is not None and args.thermalization_fraction >= 1:
        parser.error('--thermalization-fraction must be smaller than one')
    return args


def plot_results(times, temporal, projected, ks, path, title, gap_times=None, fraction=None):
    """Plot disorder-averaged temporal and projected distances.

    Parameters
    ----------
    times : real array, shape (number_of_reporting_times,)
        The saved endpoint times; time zero is omitted from logarithmic plots.
    temporal, projected : real arrays, shape (realizations, times, moment_orders)
        Temporal/RP distance and running mean of per-state projected/Scrooge
        distance, respectively, as calculated by main.
    ks : array of integers
        Moment order corresponding to each final array index.
    path : str
        Filename label appended to each figure name; not a directory path.
    title : str
        Figure title describing the run.
    gap_times : array, shape (realizations, moment_orders), or None
        Resolved spectral reference times 2*pi/minimum k-sum gap. The temporal
        plot can show their finite median; skipped/unresolved entries are NaN.
        These reference lines are not predictions of the projected timescale.
    fraction : float in (0,1), or None
        If supplied, mark observed sustained crossings below this fraction
        of the initial mean distance, separately for each curve.

    Two PNG figures are saved under utilities.FIG_DIR. Shading is the standard
    error across disorder realizations, with zero shading for a single
    realization. It is not a separate estimate of Scrooge or time-sampling
    uncertainty. This function does not recompute states or moments."""
    import matplotlib.pyplot as plt
    os.makedirs(ut.FIG_DIR, exist_ok=True)
    for data, label, prefix in (
        (temporal, r'$||\rho_{Temp}^{(k)}-\rho_{RP}^{(k)}||_2$', 'temp_rp_2norm'),
        (projected, r'$\mathbb{E}_{t}\,||\rho_{Proj,t}^{(k)}-\rho_{Scr,cond}^{(k)}||_2$', 'proj_scr_2norm'),
    ):
        mean = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1)/np.sqrt(len(data)) if len(data) > 1 else np.zeros_like(mean)
        fig, ax = plt.subplots()
        for j, k in enumerate(ks):
            ax.plot(times[1:], mean[1:, j], label=f'k={k}')
            ax.fill_between(times[1:], np.maximum(mean[1:, j]-sem[1:, j], 0), mean[1:, j]+sem[1:, j], alpha=.3)
        th.annotate_timescales(ax, times, mean, ks,
                              gap_times=gap_times if prefix == 'temp_rp_2norm' else None,
                              fraction=fraction)
        ax.set(xlabel=r'$\tau$', ylabel=label, xscale='log', yscale='log')
        ax.legend()
        fig.suptitle(title)
        fig.savefig(os.path.join(ut.FIG_DIR, prefix+'_'+path+'.png'), **ut.FIG_SAVE_OPTIONS)
        plt.close(fig)


def main():
    """Run the two ensemble comparisons for each disorder realization.

    In units with hbar=1, construct H=H_SYK4+(g/N)H_SYK2 and diagonalize it in
    even parity by default. The initial state is |0>, so its energy-basis
    coefficients are c_alpha=<E_alpha|0> and populations p_alpha=|c_alpha|**2.

    The temporal branch computes continuous-time kth-moment distances from
    spectral overlaps using temporal_distance_continuous. The projection
    branch constructs the conditional Scrooge target from the diagonal
    energy ensemble, evolves full states at separate sampled times, and
    calculates each state's projected/Scrooge distance. It then averages
    these distances over sampled times, rather than first averaging projected
    moments and subsequently taking one distance.

    For reporting interval width Delta, the stratified mode draws one
    independent uniform time in each interval [(j-1)Delta,jDelta). The mean
    of the first j distances estimates the uniform time average over
    [0,j*Delta]. Time zero is returned separately. 'grid' mode instead uses
    the discrete mean over endpoint states, including time zero. Reporting
    intervals may be slightly narrower than --projected-dt so --tmax is an
    exact endpoint. Neither rule changes the continuous temporal calculation.

    Independent disorder realizations are distributed across MPI processes;
    each process diagonalizes and analyzes its own realizations. Results are
    collected in realization order. Combined results are saved in an NPZ
    file, with times, moment orders, parameters, and distance arrays of shape
    (realizations, reporting_times, moment_orders). Per-realization JSON
    files record stage times and integration diagnostics. Some expensive
    intermediate results can be reused by their parameter-specific names.
    Optional figures show averages and standard errors across disorder.
    --gap-lines adds spectral references only to temporal plots, using saved
    minimum k-sum gaps. --thermalization-fraction marks finite-window sustained
    crossings of a fraction of each initial disorder-mean distance. Neither
    marker is assumed to be a universal equilibration time.
    Default console output is limited to rank-zero start/completion, result
    locations and a combined sampling warning if applicable. --verbose prints
    per-realization details from every rank. Detailed timings, gap diagnostics,
    integration-refinement results and warnings are always saved, irrespective
    of verbosity. Saved analysis wall time includes computation, collection and
    per-realization diagnostics, but excludes final archive writing and plotting. Cached eigensystems
    are checked before constructing H, and CACHE_DIR remains a static default."""
    args = parse_args()
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank, size = comm.Get_rank(), comm.Get_size()
    run_started = perf_counter()
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    dm.config.L = args.N//2
    D, DA = 2**dm.config.L, 2**(args.NA//2)
    subspace = ss.Parity('even', L=dm.config.L) if args.parity == 'even' else None
    dimension = D//2 if subspace is not None else D
    indices = (subspace.idx_to_state(np.arange(dimension)) if subspace is not None else np.arange(D))
    # Initial product state |0> is even. Keep the explicit mapping; compact
    # parity coordinates must never be reshaped directly into bath x subsystem.
    psi0 = np.zeros(dimension, dtype=np.complex128)
    psi0[np.flatnonzero(indices == 0)[0]] = 1
    ks = np.arange(1, args.k_max+1)
    n_intervals = max(1, int(np.ceil(args.tmax/args.projected_dt)))
    times = np.linspace(0, args.tmax, n_intervals+1)
    width = times[1]
    run_note = (f'N{args.N}_g{args.g}_parity{args.parity}_seed{args.seed}'
                f'_R{args.realizations}_DA{DA}_k{args.k_max}_T{args.tmax}'
                f'_dt{args.dt}_pdt{args.projected_dt}_ps{args.projected_sampling}'
                f'_ts{args.time_seed}_scr{args.n_scr}_check{int(args.check_dt)}'
                f'_gap{args.gap_k_max}_gcap{args.gap_max_sums}'
                f'_fraction{args.thermalization_fraction}')
    task_alloc = [np.arange(i, args.realizations, size) for i in range(size)]
    tasks = task_alloc[rank]
    local_tr = np.empty((len(tasks), len(times), len(ks)))
    local_ps = np.empty_like(local_tr)
    local_gaps = np.full((len(tasks), len(ks)), np.nan)
    local_gap_times = np.full_like(local_gaps, np.nan)
    local_sampling_warnings = np.zeros(len(tasks), dtype=np.int64)
    if rank == 0:
        print(f'{args.realizations} realizations on {size} ranks; sector dimension {dimension}; '
              f'{n_intervals} projected strata; temporal dt={args.dt}', flush=True)
    for i, r in enumerate(tasks):
        rng = lambda stream: np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(int(r), stream)))
        h_note = f'SYK42_N{args.N}_g{args.g}_base-seed{args.seed}_r{r}_parity{args.parity}'
        scr_note = f'{h_note}_psi00_DA{DA}_scr{args.n_scr}_k{args.k_max}_sym'
        t0 = perf_counter()
        eig_hit = args.cache_eigensystem and la.eig_system.cache_exists(note=h_note, cache_dir=cache_dir)
        if eig_hit:
            # The eigensystem is all the analysis needs; no H is loaded or built.
            t1 = perf_counter()
            eigs = la.eig_system(None, note=h_note, cache_dir=cache_dir, require_cache=True)
        else:
            H = hf.SYK42_numpy(args.g, rng(0), subspace=subspace, note=h_note,
                              cache_dir=cache_dir, cache_enabled=args.cache_hamiltonian)
            t1 = perf_counter()
            eigs = la.eig_system(H, note=h_note, cache_dir=cache_dir,
                                 cache_enabled=args.cache_eigensystem)
            del H
        t2 = perf_counter()
        evals, evecs = eigs['evals'], eigs['evecs']
        coefficients = evecs.conj().T @ psi0
        pops = np.abs(coefficients)**2
        max_dt = tf.spectral_sampling_diagnostics(evals, args.k_max, 1.)['nyquist_dt']
        dt = min(.1, .5*max_dt) if args.dt == 'auto' else args.dt
        diagnostics = tf.spectral_sampling_diagnostics(evals, args.k_max, dt)
        diagnostics['warnings'] = []
        if args.verbose:
            print(f'Rank {rank}, realization {r}: bandwidth={diagnostics["bandwidth"]:.6g}, '
                  f'Nyquist bound dt<{max_dt:.6g}, using dt<={dt:.6g}', flush=True)
        if dt >= max_dt:
            warning = ('Temporal grid exceeds the conservative moment-frequency bound; '
                       'use --dt auto or verify with --check-dt')
            diagnostics['warnings'].append(warning)
            local_sampling_warnings[i] = 1
            if args.verbose:
                print(f'WARNING: rank {rank}, realization {r}: {warning}', flush=True)
        temporal = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt)
        local_tr[i] = temporal['distance']
        diagnostics['dt_effective'] = float(temporal['dt_effective'])
        centered_energies = evals-evals[0]
        mean_energy = float(pops @ centered_energies)
        energy_variance = float(pops @ (centered_energies-mean_energy)**2)
        diagnostics['energy_standard_deviation'] = float(np.sqrt(energy_variance))
        diagnostics['short_time_overlap_widths'] = [
            float(1/np.sqrt(k*energy_variance)) if energy_variance > 0 else None for k in ks]
        if args.check_dt:
            refined = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt/2)
            diagnostics['dt_half_max_squared_error'] = np.max(np.abs(
                temporal['distance_squared']-refined['distance_squared']), axis=0).tolist()
            diagnostics['dt_half_max_distance_error'] = np.max(np.abs(
                temporal['distance']-refined['distance']), axis=0).tolist()
            if args.verbose:
                print(f'Rank {rank}, dt refinement: {diagnostics["dt_half_max_distance_error"]}', flush=True)
        gap_start = perf_counter()
        diagnostics['moment_gaps'] = []
        occupied = evals[pops > 0]  # omit only exactly zero populations
        for j, k in enumerate(ks):
            if k <= args.gap_k_max:
                gap = th.minimum_moment_gap(occupied, int(k), args.gap_max_sums)
                gap['k'] = int(k)
                diagnostics['moment_gaps'].append(gap)
                if gap['min_gap'] is not None:
                    local_gaps[i, j] = gap['min_gap']
                if gap['reference_time'] is not None:
                    local_gap_times[i, j] = gap['reference_time']
                if args.verbose:
                    print(f'Rank {rank}, k={k} gap diagnostic: {gap}', flush=True)
        t3 = perf_counter()
        target = pf.cond_scr_moment(evecs, pops, args.n_scr, ks, DA, rng(1),
                    note=scr_note, basis_indices=indices, full_dimension=D, symmetric=True, cache_dir=cache_dir)
        t4 = perf_counter()
        sample_times = times.copy()
        if args.projected_sampling == 'stratified':
            time_rng = np.random.default_rng(np.random.SeedSequence(
                args.seed, spawn_key=(int(r), 2, args.time_seed)))
            sample_times[1:] = times[:-1] + width*time_rng.random(n_intervals)
        sampled = pf.time_evolved_overlaps_and_ps_2norm(
            evals, evecs, coefficients, sample_times, ks, DA, target,
            chunk=args.chunk, note=f'{run_note}_r{r}', basis_indices=indices,
            full_dimension=D, symmetric=True, cache_dir=cache_dir,
            cache_enabled=args.cache_projected)
        distances = sampled['2norm']
        if args.projected_sampling == 'stratified':
            local_ps[i, 0] = distances[0]
            local_ps[i, 1:] = np.cumsum(distances[1:], axis=0)/np.arange(1, n_intervals+1)[:, None]
        else:
            # Explicitly retain the old discrete ensemble convention for comparison.
            local_ps[i] = np.cumsum(distances, axis=0)/np.arange(1, len(times)+1)[:, None]
        t5 = perf_counter()
        timing = {'hamiltonian_seconds': t1-t0, 'diagonalization_seconds': t2-t1,
                  'temporal_seconds': gap_start-t2, 'gap_seconds': t3-gap_start, 'scrooge_seconds': t4-t3,
                  'projected_seconds': t5-t4, 'total_seconds': t5-t0,
                  'eigensystem_cache_hit': eig_hit, 'rank': rank, 'realization': int(r), 'parameters': vars(args),
                  'sampling_diagnostics': diagnostics}
        with open(os.path.join(cache_dir, f'timing_{run_note}_r{r}.json'), 'w') as f:
            json.dump(timing, f, indent=2)
        if args.verbose:
            print(f'Rank {rank}, realization {r}: timings {timing["total_seconds"]:.1f}s total, '
                  f'{timing["projected_seconds"]:.1f}s projected', flush=True)
    # Gather contiguous rank blocks, then restore disorder-realization order.
    counts = np.array([len(x) for x in task_alloc])*len(times)*len(ks)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    temporal = np.empty((args.realizations, len(times), len(ks))) if rank == 0 else None
    projected = np.empty_like(temporal) if rank == 0 else None
    comm.Gatherv(local_tr, [temporal, (counts, offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_ps, [projected, (counts, offsets)] if rank == 0 else None, root=0)
    gap_counts = np.array([len(x) for x in task_alloc])*len(ks)
    gap_offsets = np.r_[0, np.cumsum(gap_counts[:-1])]
    gaps = np.empty((args.realizations, len(ks))) if rank == 0 else None
    gap_times = np.empty_like(gaps) if rank == 0 else None
    comm.Gatherv(local_gaps, [gaps, (gap_counts, gap_offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_gap_times, [gap_times, (gap_counts, gap_offsets)] if rank == 0 else None, root=0)
    realization_counts = np.array([len(x) for x in task_alloc])
    realization_offsets = np.r_[0, np.cumsum(realization_counts[:-1])]
    sampling_warnings = np.empty(args.realizations, dtype=np.int64) if rank == 0 else None
    comm.Gatherv(local_sampling_warnings,
                 [sampling_warnings, (realization_counts, realization_offsets)] if rank == 0 else None, root=0)
    if rank == 0:
        order = np.argsort(np.concatenate(task_alloc))
        temporal, projected = temporal[order], projected[order]
        gaps, gap_times = gaps[order], gap_times[order]
        sampling_warnings = sampling_warnings[order]
        analysis_wall_seconds = perf_counter()-run_started
        result_path = os.path.join(cache_dir, 'results_'+run_note+'.npz')
        def observed_times(data):
            """Threshold endpoints for disorder-mean curves; NaN denotes no crossing."""
            if args.thermalization_fraction is None:
                return np.full(len(ks), np.nan)
            mean = data.mean(axis=0)
            return np.array([np.nan if (t := th.threshold_time(times, mean[:, j],
                args.thermalization_fraction)) is None else t for j in range(len(ks))])
        np.savez(result_path, tau_series=times, k_series=ks,
                 temp_rp_2norm=temporal, proj_scr_2norm=projected,
                 moment_min_gaps=gaps, moment_gap_times=gap_times,
                 sampling_bound_warnings=sampling_warnings,
                 analysis_wall_seconds=np.array(analysis_wall_seconds),
                 temporal_threshold_times=observed_times(temporal),
                 projected_threshold_times=observed_times(projected),
                 metadata=np.array(json.dumps(vars(args))))
        print(f'Analysis complete: {args.realizations} realizations in {analysis_wall_seconds:.1f}s', flush=True)
        if sampling_warnings.any():
            print(f'WARNING: {int(sampling_warnings.sum())}/{args.realizations} realizations exceed the '
                  'temporal sampling bound; see saved diagnostics.', flush=True)
        print(f'Results saved: {result_path}', flush=True)
        print(f'Detailed timings and diagnostics: {os.path.join(cache_dir, "timing_"+run_note+"_r*.json")}', flush=True)
        if not args.no_plots:
            plot_results(times, temporal, projected, ks, run_note,
                         f'SYK42 N={args.N}, g={args.g}, R={args.realizations}, parity={args.parity}',
                         gap_times=gap_times if args.gap_lines else None,
                         fraction=args.thermalization_fraction)


if __name__ == '__main__':
    main()
