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


def validate_N(arg):
    num = int(arg)
    if num <= 4 or num % 2:
        raise argparse.ArgumentTypeError('N must be an even number of Majoranas greater than 4')
    return num


def positive_float(arg):
    value = float(arg)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return value


def positive_int(arg):
    value = int(arg)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', type=validate_N, required=True)
    parser.add_argument('--g', type=float, required=True)
    parser.add_argument('--dt', default='auto', help='Temporal quadrature max step: auto or a positive number')
    parser.add_argument('--projected-dt', type=positive_float, default=1., help='Projected stratum width; independent of --dt')
    parser.add_argument('--projected-sampling', choices=('stratified', 'grid'), default='stratified')
    parser.add_argument('--tmax', type=positive_float, default=1e5)
    parser.add_argument('--parity', choices=('even', 'full'), default='even')
    parser.add_argument('--realizations', type=positive_int, default=128, help='Independent of MPI rank count')
    parser.add_argument('--n-scr', type=positive_int, default=1000)
    parser.add_argument('--k-max', type=positive_int, default=3)
    parser.add_argument('--NA', type=positive_int, default=4)
    parser.add_argument('--chunk', type=positive_int, default=32, help='Projected time batch size')
    parser.add_argument('--seed', type=int, default=628**3)
    parser.add_argument('--time-seed', type=int, default=0, help='Independent stratified sampling replicate')
    parser.add_argument('--check-dt', action='store_true', help='Also compute dt/2 and record convergence error')
    parser.add_argument('--cache-dir', default=ut.CACHE_DIR)
    parser.add_argument('--no-plots', action='store_true')
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
    return args


def plot_results(times, temporal, projected, ks, path, title):
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
        ax.set(xlabel=r'$\tau$', ylabel=label, xscale='log', yscale='log')
        ax.legend()
        fig.suptitle(title)
        fig.savefig(os.path.join(ut.FIG_DIR, prefix+'_'+path+'.png'), **ut.FIG_SAVE_OPTIONS)
        plt.close(fig)


def main():
    args = parse_args()
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank, size = comm.Get_rank(), comm.Get_size()
    ut.CACHE_DIR = args.cache_dir
    os.makedirs(ut.CACHE_DIR, exist_ok=True)
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
    run_note = (f'v2_N{args.N}_g{args.g}_parity{args.parity}_seed{args.seed}'
                f'_R{args.realizations}_DA{DA}_k{args.k_max}_T{args.tmax}'
                f'_dt{args.dt}_pdt{args.projected_dt}_ps{args.projected_sampling}'
                f'_ts{args.time_seed}_scr{args.n_scr}_check{int(args.check_dt)}')
    task_alloc = [np.arange(i, args.realizations, size) for i in range(size)]
    tasks = task_alloc[rank]
    local_tr = np.empty((len(tasks), len(times), len(ks)))
    local_ps = np.empty_like(local_tr)
    if rank == 0:
        print(f'{args.realizations} realizations on {size} ranks; sector dimension {dimension}; '
              f'{n_intervals} projected strata; temporal dt={args.dt}', flush=True)
    for i, r in enumerate(tasks):
        rng = lambda stream: np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(int(r), stream)))
        h_note = f'SYK42_N{args.N}_g{args.g}_base-seed{args.seed}_r{r}_parity{args.parity}'
        scr_note = f'{h_note}_psi00_DA{DA}_scr{args.n_scr}_k{args.k_max}_sym_v2'
        t0 = perf_counter()
        H = hf.SYK42_numpy(args.g, rng(0), subspace=subspace, note=h_note)
        t1 = perf_counter()
        eigs = la.eig_system(H, note=h_note)
        del H
        t2 = perf_counter()
        evals, evecs = eigs['evals'], eigs['evecs']
        coefficients = evecs.conj().T @ psi0
        pops = np.abs(coefficients)**2
        max_dt = tf.spectral_sampling_diagnostics(evals, args.k_max, 1.)['nyquist_dt']
        dt = min(.1, .5*max_dt) if args.dt == 'auto' else args.dt
        diagnostics = tf.spectral_sampling_diagnostics(evals, args.k_max, dt)
        print(f'Rank {rank}, realization {r}: bandwidth={diagnostics["bandwidth"]:.6g}, '
              f'Nyquist bound dt<{max_dt:.6g}, using dt<={dt:.6g}', flush=True)
        if dt >= max_dt:
            print(f'WARNING: rank {rank}: temporal grid exceeds conservative moment-frequency bound; '
                  'use --dt auto or verify with --check-dt', flush=True)
        temporal = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt)
        local_tr[i] = temporal['distance']
        diagnostics['dt_effective'] = float(temporal['dt_effective'])
        if args.check_dt:
            refined = tf.temporal_distance_continuous(evals, pops, ks, times, dt=dt/2)
            diagnostics['dt_half_max_squared_error'] = np.max(np.abs(
                temporal['distance_squared']-refined['distance_squared']), axis=0).tolist()
            diagnostics['dt_half_max_distance_error'] = np.max(np.abs(
                temporal['distance']-refined['distance']), axis=0).tolist()
            print(f'Rank {rank}, dt refinement: {diagnostics["dt_half_max_distance_error"]}', flush=True)
        t3 = perf_counter()
        target = pf.cond_scr_moment(evecs, pops, args.n_scr, ks, DA, rng(1),
                    note=scr_note, basis_indices=indices, full_dimension=D, symmetric=True)
        t4 = perf_counter()
        sample_times = times.copy()
        if args.projected_sampling == 'stratified':
            time_rng = np.random.default_rng(np.random.SeedSequence(
                args.seed, spawn_key=(int(r), 2, args.time_seed)))
            sample_times[1:] = times[:-1] + width*time_rng.random(n_intervals)
        sampled = pf.time_evolved_overlaps_and_ps_2norm(
            evals, evecs, coefficients, sample_times, ks, DA, target,
            chunk=args.chunk, note=f'{run_note}_r{r}', basis_indices=indices,
            full_dimension=D, symmetric=True)
        distances = sampled['2norm']
        if args.projected_sampling == 'stratified':
            local_ps[i, 0] = distances[0]
            local_ps[i, 1:] = np.cumsum(distances[1:], axis=0)/np.arange(1, n_intervals+1)[:, None]
        else:
            # Explicitly retain the old discrete ensemble convention for comparison.
            local_ps[i] = np.cumsum(distances, axis=0)/np.arange(1, len(times)+1)[:, None]
        t5 = perf_counter()
        timing = {'hamiltonian_seconds': t1-t0, 'diagonalization_seconds': t2-t1,
                  'temporal_seconds': t3-t2, 'scrooge_seconds': t4-t3,
                  'projected_seconds': t5-t4, 'total_seconds': t5-t0,
                  'rank': rank, 'realization': int(r), 'parameters': vars(args),
                  'sampling_diagnostics': diagnostics}
        with open(os.path.join(ut.CACHE_DIR, f'timing_{run_note}_r{r}.json'), 'w') as f:
            json.dump(timing, f, indent=2)
        print(f'Rank {rank}, realization {r}: timings {timing["total_seconds"]:.1f}s total, '
              f'{timing["projected_seconds"]:.1f}s projected', flush=True)
    # Gather contiguous rank blocks, then restore disorder-realization order.
    counts = np.array([len(x) for x in task_alloc])*len(times)*len(ks)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    temporal = np.empty((args.realizations, len(times), len(ks))) if rank == 0 else None
    projected = np.empty_like(temporal) if rank == 0 else None
    comm.Gatherv(local_tr, [temporal, (counts, offsets)] if rank == 0 else None, root=0)
    comm.Gatherv(local_ps, [projected, (counts, offsets)] if rank == 0 else None, root=0)
    if rank == 0:
        order = np.argsort(np.concatenate(task_alloc))
        temporal, projected = temporal[order], projected[order]
        result_path = os.path.join(ut.CACHE_DIR, 'results_'+run_note+'.npz')
        np.savez(result_path, tau_series=times, k_series=ks,
                 temp_rp_2norm=temporal, proj_scr_2norm=projected,
                 metadata=np.array(json.dumps(vars(args))))
        print(f'Results saved: {result_path}', flush=True)
        if not args.no_plots:
            plot_results(times, temporal, projected, ks, run_note,
                         f'SYK42 N={args.N}, g={args.g}, R={args.realizations}, parity={args.parity}')


if __name__ == '__main__':
    main()
