"""(1-g) SYK4 + g SYK2: continuous temporal and sampled projected ensembles."""
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
import ensemble_analysis as ea
from ensemble_analysis import positive_float, positive_int, plot_results


def validate_N(arg):
    """Read N as an even number of Majorana modes greater than four.

    The spin representation has N/2 two-level sites, so odd N is rejected
    rather than silently replaced by N-1. Used to validate the --N argument."""
    num = int(arg)
    if num <= 4 or num % 2:
        raise argparse.ArgumentTypeError('N must be an even number of Majoranas greater than 4')
    return num


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
    parser.add_argument('--g', type=float, required=True, help='Dimensionless slider in (1-g)H4+g H2, between zero and one')
    ea.add_sampling_arguments(parser)
    parser.add_argument('--parity', choices=('even', 'full'), default='even', help='Eigenvector basis; even is valid for the vacuum initial state')
    parser.add_argument('--realizations', type=positive_int, default=128, help='Independent of MPI rank count')
    parser.add_argument('--NA', type=positive_int, default=4, help='Even subsystem Majorana count; DA=2**(NA/2)')
    args = parser.parse_args()
    if args.NA % 2 or args.NA > args.N:
        parser.error('NA must be even and no larger than N')
    if not np.isfinite(args.g) or not 0 <= args.g <= 1 or args.seed < 0 or args.time_seed < 0:
        parser.error('g must lie between zero and one; seeds must be nonnegative')
    ea.validate_sampling_arguments(parser,args)
    return args


def main():
    """Run four distance curves, including the unaveraged single-Scrooge distance.

    In units with hbar=1, construct H=(1-g)H_SYK4+g H_SYK2 and diagonalize it in
    even parity by default. The initial state is |0>, so its energy-basis
    coefficients are c_alpha=<E_alpha|0> and populations p_alpha=|c_alpha|**2.

    The temporal branch computes continuous-time kth-moment distances from
    spectral overlaps using temporal_distance_continuous. The projection
    branch constructs the conditional Scrooge target from the diagonal
    energy ensemble, evolves full states at separate sampled times, and
    calculates each state's projected/Scrooge distance. The same evolved state
    also supplies rho_A(t) for the single Scrooge target. The single-target
    first-moment distance is evaluated numerically even though it is
    mathematically zero. It then averages
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
    # Set up MPI, the physical Hilbert space, and the initial ket.
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
    # Report the actual complex initial amplitudes in the computational basis,
    # independently of any later choice of energy eigenvector phases.
    nonzero_initial = np.flatnonzero(psi0)
    initial_state_description = {
        'basis': 'full computational basis',
        'basis_labels': indices[nonzero_initial].tolist(),
        'amplitudes_real': psi0[nonzero_initial].real.tolist(),
        'amplitudes_imag': psi0[nonzero_initial].imag.tolist()}

    # Reporting endpoints are shared by both comparisons. The temporal
    # integration uses a finer grid; projected states have separate time draws.
    ks = np.arange(1, args.k_max+1)
    n_intervals = max(1, int(np.ceil(args.tmax/args.projected_dt)))
    times = np.linspace(0, args.tmax, n_intervals+1)
    width = times[1]
    run_note = (f'mix_N{args.N}_g{args.g}_parity{args.parity}_seed{args.seed}'
                f'_R{args.realizations}_DA{DA}_k{args.k_max}_T{args.tmax}'
                f'_dt{args.dt}_pdt{args.projected_dt}_ps{args.projected_sampling}'
                f'_ts{args.time_seed}_scr{args.n_scr}_check{int(args.check_dt)}'
                f'_gap{args.gap_k_max}_gcap{args.gap_max_sums}'
                f'_fraction{args.thermalization_fraction}'
                f'_rpenergyblocks_tol{args.degeneracy_tolerance}_singlecheck')
    # Distribute disorder realizations across MPI ranks. Each local result
    # array has axes (assigned realizations, reporting times, moment orders).
    task_alloc = [np.arange(i, args.realizations, size) for i in range(size)]
    tasks = task_alloc[rank]
    local_tr = np.empty((len(tasks), len(times), len(ks)))
    local_ps = np.empty_like(local_tr)
    local_single = np.empty_like(local_tr)
    local_single_instant = np.empty_like(local_tr)
    local_sample_times = np.empty((len(tasks),len(times)))
    local_gaps = np.full((len(tasks), len(ks)), np.nan)
    local_gap_times = np.full_like(local_gaps, np.nan)
    local_sampling_warnings = np.zeros(len(tasks), dtype=np.int64)
    if rank == 0:
        print(f'{args.realizations} realizations on {size} ranks; sector dimension {dimension}; '
              f'{n_intervals} projected strata; temporal dt={args.dt}', flush=True)
    for i, r in enumerate(tasks):
        # Begin one disorder realization. Separate random streams control the
        # Hamiltonian, fixed Scrooge samples, and instantaneous Scrooge samples.
        rng = lambda stream: np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(int(r), stream)))
        h_note = f'SYK42mix_N{args.N}_g{args.g}_base-seed{args.seed}_r{r}_parity{args.parity}'
        scr_note = (f'{h_note}_psi00_DA{DA}_scr{args.n_scr}_k{args.k_max}_sym'
                    f'_rpenergyblocks_tol{args.degeneracy_tolerance}')
        # 1. Obtain the Hamiltonian's energies and eigenvectors, loading the
        # raw eigensystem from cache when available.
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

        # 2. Adapt the eigenbasis to psi0 ONCE. Each occupied degenerate energy
        # block now has one occupied ket; all other coefficients in it are zero.
        # Phases are retained in the complex basis kets: evecs@coefficients
        # reconstructs the original psi0. No dephasing is applied to that ket.
        # Every subsequent calculation uses this same adapted eigensystem.
        adapted = ea.prepare_eigensystem(eigs,psi0,ks,args)
        evals, evecs = adapted['evals'], adapted['evecs']
        coefficients, pops = adapted['coefficients'], adapted['pops']
        dt, diagnostics = adapted['dt'], adapted['diagnostics']
        max_dt = diagnostics['nyquist_dt']
        active = pops > 0  # Omit exactly empty columns, with no further rotation.
        del eigs, adapted

        # 3. Begin the continuous temporal-versus-RP moment comparison.
        # Basis and sampling diagnostics are prepared above without dephasing psi0.
        if args.verbose:
            print(f'Rank {rank}, realization {r}: bandwidth={diagnostics["bandwidth"]:.6g}, '
                  f'Nyquist bound dt<{max_dt:.6g}, using dt<={dt:.6g}', flush=True)
        if diagnostics['warnings']:
            local_sampling_warnings[i] = 1
            if args.verbose:
                for warning in diagnostics['warnings']:
                    print(f'WARNING: rank {rank}, realization {r}: {warning}',flush=True)

        # Compute the temporal moment errors from scalar return amplitudes;
        # temporal_distance_continuous also computes the exact RP purities.
        temporal = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt)
        local_tr[i] = temporal['distance']

        # Record checks and short-time energy scales for the temporal result.
        ea.temporal_checks(evals,pops,ks,local_tr[i],diagnostics)
        diagnostics['dt_effective'] = float(temporal['dt_effective'])

        # Optional integration-accuracy check: repeat only the scalar-overlap
        # calculation with half the step. No projected states are generated here.
        if args.check_dt:
            refined = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt/2)
            diagnostics['dt_half_max_squared_error'] = np.max(np.abs(
                temporal['distance_squared']-refined['distance_squared']), axis=0).tolist()
            diagnostics['dt_half_max_distance_error'] = np.max(np.abs(
                temporal['distance']-refined['distance']), axis=0).tolist()
            if args.verbose:
                print(f'Rank {rank}, dt refinement: {diagnostics["dt_half_max_distance_error"]}', flush=True)

        # Optional spectral timescale diagnostics, separate from the moment
        # integration. Enumerate k-energy sums only when the requested cap allows.
        gap_start = perf_counter()
        diagnostics['moment_gaps'] = []
        occupied = evals[active]
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

        # 4. Compute the fixed generalized Scrooge target G_k[sigma]. In this
        # adapted basis sigma = sum_alpha pops[alpha] |E_alpha><E_alpha| is
        # already the physical energy-dephased state. Pass its occupied kets
        # and populations directly; no second dephasing or basis adaptation.
        target = pf.cond_scr_moment(
            evecs[:, active], pops[active], args.n_scr, ks, DA, rng(1),
            note=scr_note+'_dephased_tol0.0',
            basis_indices=indices, full_dimension=D, symmetric=True, cache_dir=cache_dir)
        t4 = perf_counter()

        # 5. Begin the projected-ensemble computation. Choose a time per
        # interval (or grid endpoints), evolve psi0 using the adapted basis,
        # and compute each state's projected moments and target distances.
        sample_times = ea.projected_times(times,args.projected_sampling,args.seed,
                                         args.time_seed,int(r))
        sampled = pf.time_evolved_overlaps_and_ps_2norm(
            evals[active], evecs[:, active], coefficients[active], sample_times, ks, DA, target,
            chunk=args.chunk, note=f'{run_note}_r{r}', basis_indices=indices,
            full_dimension=D, symmetric=True, cache_dir=cache_dir,
            cache_enabled=args.cache_projected, instantaneous=True, n_scr=args.n_scr, rng=rng(3))

        # 6. Average the per-state distances over sampled times. This differs
        # from averaging projected moments before calculating a distance.
        # The same evolved states also supply the single Scrooge(rho_A(t))
        # comparison, including its numerically evaluated k=1 distance.
        distances = sampled['2norm']
        local_ps[i] = ea.running_distance(distances,args.projected_sampling)
        single_distances = sampled['single_scr_2norm']
        # Reuse every per-state distance before the cumulative time average.
        # In stratified mode its time is sample_times[j], not endpoint times[j].
        local_single_instant[i] = single_distances
        local_sample_times[i] = sample_times
        local_single[i] = ea.running_distance(single_distances,args.projected_sampling)
        diagnostics['instantaneous_targets'] = {
            'single_scrooge': 'rho_A(t), analytic k=1 and rank-one inputs; independent Haar samples otherwise',
            'n_scr_per_mixed_state': args.n_scr,
            'averaging': 'same sampled times and distance-before-time-average convention as the fixed target'}
        t5 = perf_counter()

        # Save this realization's timings and diagnostics on its assigned rank.
        # The distance arrays remain local until every assigned realization ends.
        timing = {'hamiltonian_seconds': t1-t0, 'diagonalization_seconds': t2-t1,
                  'temporal_seconds': gap_start-t2, 'gap_seconds': t3-gap_start, 'scrooge_seconds': t4-t3,
                  'projected_seconds': t5-t4, 'total_seconds': t5-t0,
                  'eigensystem_cache_hit': eig_hit, 'rank': rank, 'realization': int(r), 'parameters': vars(args), 'hamiltonian_convention': '(1-g)*SYK4+g*SYK2',
                  'initial_state': initial_state_description,
                  'sampling_diagnostics': diagnostics}
        with open(os.path.join(cache_dir, f'timing_{run_note}_r{r}.json'), 'w') as f:
            json.dump(timing, f, indent=2)
        if args.verbose:
            print(f'Rank {rank}, realization {r}: timings {timing["total_seconds"]:.1f}s total, '
                  f'{timing["projected_seconds"]:.1f}s projected', flush=True)
    # 7. Collect all local MPI arrays on rank zero, beginning with the four
    # distance arrays. Ranks may own different numbers of realizations.
    counts = np.array([len(x) for x in task_alloc])*len(times)*len(ks)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    temporal = np.empty((args.realizations, len(times), len(ks))) if rank == 0 else None
    projected = np.empty_like(temporal) if rank == 0 else None
    single = np.empty_like(temporal) if rank == 0 else None
    single_instant = np.empty_like(temporal) if rank == 0 else None
    projected_sample_times = np.empty((args.realizations,len(times))) if rank == 0 else None
    comm.Gatherv(local_tr, [temporal, (counts, offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_ps, [projected, (counts, offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_single, [single, (counts, offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_single_instant, [single_instant, (counts, offsets)] if rank == 0 else None, root=0)
    time_counts = np.array([len(x) for x in task_alloc])*len(times)
    time_offsets = np.r_[0,np.cumsum(time_counts[:-1])]
    comm.Gatherv(local_sample_times,
        [projected_sample_times,(time_counts,time_offsets)] if rank == 0 else None,root=0)

    # Collect the smaller arrays of spectral references and sampling warnings.
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

    # 8. Rank zero restores realization order, saves the combined results,
    # and optionally plots disorder averages. Other ranks have finished.
    if rank == 0:
        order = np.argsort(np.concatenate(task_alloc))
        temporal, projected = temporal[order], projected[order]
        single = single[order]
        single_instant = single_instant[order]
        projected_sample_times = projected_sample_times[order]
        gaps, gap_times = gaps[order], gap_times[order]
        sampling_warnings = sampling_warnings[order]
        analysis_wall_seconds = perf_counter()-run_started
        result_path = os.path.join(cache_dir, 'results_'+run_note+'.npz')

        def observed_times(data):
            return ea.observed_times(times,data,ks,args.thermalization_fraction)

        np.savez_compressed(result_path, tau_series=times, k_series=ks,
                 temp_rp_2norm=temporal, proj_scr_2norm=projected,
                 proj_single_scr_2norm=single,
                 proj_single_scr_instant_2norm=single_instant,
                 projected_sample_times=projected_sample_times,
                 moment_min_gaps=gaps, moment_gap_times=gap_times,
                 sampling_bound_warnings=sampling_warnings,
                 analysis_wall_seconds=np.array(analysis_wall_seconds),
                 temporal_threshold_times=observed_times(temporal),
                 projected_threshold_times=observed_times(projected),
                 single_scrooge_threshold_times=observed_times(single),
                 metadata=np.array(json.dumps(dict(vars(args), hamiltonian_convention='(1-g)*SYK4+g*SYK2',
                     rp_reference='initial-state-adapted energy blocks',
                     initial_state=initial_state_description,
                     single_scrooge_evaluation='numerical moment difference',
                     energy_grouping='mean energy within each numerically degenerate block'))))
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
                         fraction=args.thermalization_fraction, single=single,
                         single_instant=single_instant,sample_times=projected_sample_times)


if __name__ == '__main__':
    main()
