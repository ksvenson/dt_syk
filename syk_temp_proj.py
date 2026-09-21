import dynamite as dm
import dynamite.computations as comp
import dynamite.operators as op
import dynamite.states as st
import dynamite.subspaces as ss
import dynamite.tools as tools
import numpy as np
import os
import argparse
# matplotlib.pyplot gets imported on rank 0 later on

import linalg as la
import hamiltonian_factory as hf
import temporal_funcs as tf
import projected_funcs as pf
import random_phase as rp
import utilities as ut


def validate_N(arg):
    num = int(arg)
    if num <= 4:
        raise ValueError('N must be greater than 4.')
    return num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=validate_N, required=True, help='Total number of fermions')
    args = parser.parse_args()

    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # SYK Parameters
    N = args.N           # total number of fermions
    NA = 4               # number of fermions in subsystem
    g = 100              # coupling to 2-body interactions
    realizations = size  # disorder realizations

    # RNG config
    base_seed = 628**3  # the better circle constant!
    # s=0 for SYK couplings
    # s=1 for random unitaries
    rng = lambda r, s: np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(r, s)))
    H_title = 'SYK42'
    H_str = lambda r: f'{H_title}_N{N}_g{g}_base-seed{base_seed}_r{r}'
    H_func = hf.SYK42_numpy
    H_args = lambda r: [g, rng(r, 0)]

    # dynamite config
    dm.config.L = N // 2
    D = 2**dm.config.L
    DA = 2**(NA // 2)

    # Experiment Parameters
    k_series = np.arange(3) + 1
    n_scr = 1000  # number of scrooge states to sample
    psi0_str = '0'
    psi0 = st.State(0)
    psi0_np = psi0.to_numpy(to_all=True)
    assert psi0_np is not None  # to silence my LSP
    dt = 1.0
    log_tmax = 4
    tau_series = dt * np.arange(0, int(10**log_tmax / dt))
    temp_note = lambda r, k: f'{H_str(r)}_psi0{psi0_str}_k{k}_dt{dt}_logtmax{log_tmax}'
    proj_note = lambda r, k: f'{temp_note(r, k)}_DA{DA}'
    scr_note = lambda r, k: f'{H_str(r)}_psi0{psi0_str}_k{k}_nscr{n_scr}_DA{DA}'
    proj_scr_note = lambda r, k: f'{proj_note(r, k)}_nscr{n_scr}'
    p_str = '\n'.join((
        'Parameters:',
        rf'Hamiltonian: {H_str(realizations)}',
        f'Disorder Realizations: {realizations}',
        f'Scrooge Samples: {n_scr}',
        f'Base Seed: {base_seed}',
        rf'$|\psi_0\rangle$: {psi0_str}',
        f'dt: {dt}',
        f'DA: {DA}'
    ))

    # Task allocation across ranks
    task_alloc = [np.arange(rank, realizations, size) for rank in range(size)]
    task_counts = np.array([tasks.size for tasks in task_alloc])
    cum_task_counts = np.zeros(size, dtype=np.int32)
    cum_task_counts[1:] = np.cumsum(task_counts[:-1])

    # "tr" short for (temporal - random phase) 2-norm
    local_tr = np.empty((task_counts[rank], tau_series.size, k_series.size), dtype=np.float64)
    # "ps" short for (projected - scrooge) 2-norm
    local_ps = np.empty_like(local_tr)
    
    # Computations
    for i, r in enumerate(task_alloc[rank]):  # r stands for realization
        if rank == 0:
            print(f'Rank {rank}: {i+1} / {task_counts[rank]}')
        H_np = H_func(*H_args(r), note=H_str(r))
        if rank == 0:
            print(f'Starting Diagonalization')
        eigs = la.eig_system(H_np, note=H_str(r))
        evals = eigs['evals']
        evecs = eigs['evecs']

        psi0_eng_basis = np.einsum('ab,b->a', evecs.conj().T, psi0_np)
        pops = np.abs(psi0_eng_basis)**2
        
        if rank == 0:
            print(f'Finding conditional scrooge moment')
        rho_scr = pf.cond_scr_moment(
            evecs,
            pops,
            n_scr,
            k_series,
            DA,
            rng(r, 1),
            note=scr_note(r, f'{k_series[0]}-{k_series[-1]}')
        )
        if rank == 0:
            print(f'Finding Overlaps and Trace Distances')
        overlaps_ps = pf.time_evolved_overlaps_and_ps_2norm(
            evals,
            evecs,
            psi0_eng_basis,
            tau_series,
            k_series,
            DA,
            rho_scr,
            verbose=(rank == 0),
            note=proj_scr_note(r, f'{k_series[0]}-{k_series[-1]}')
        )
        local_ps[i, :, :] = overlaps_ps['2norm']

        local_tr[i] = tf.temp_square_2norm_sampled(overlaps_ps['overlaps'], k_series)
        local_tr[i] = local_tr[i, :, :] - rp.rpe_square_2norm(pops, k_series[-1])[np.newaxis, :]
    # take square root for the 2-norm  
    local_tr = np.sqrt(local_tr)
    # average over states drawn from the temporal ensemble
    local_ps = np.cumsum(local_ps, axis=1) / (1+np.arange(tau_series.size))[np.newaxis, :, np.newaxis]

    # Gathering rank computations
    temp_rp_2norm = None
    proj_scr_2norm = None
    tr_recv_buf = None
    ps_recv_buf = None
    if rank == 0:
        temp_rp_2norm = np.empty((realizations, *local_tr.shape[1:]), dtype=np.float64)
        proj_scr_2norm = np.empty((realizations, *local_ps.shape[1:]), dtype=np.float64)
        sizing = (
            np.prod(local_tr.shape[1:]) * task_counts,     # size of each local array
            np.prod(local_tr.shape[1:]) * cum_task_counts  # displacements
        )
        tr_recv_buf = [
            temp_rp_2norm,
            sizing
        ]
        ps_recv_buf = [
            proj_scr_2norm,
            sizing
        ]
    comm.Gatherv(local_tr, tr_recv_buf, root=0)
    comm.Gatherv(local_ps, ps_recv_buf, root=0)

    # Final statistics and plotting
    if rank == 0:
        import matplotlib.pyplot as plt
        assert temp_rp_2norm is not None  # to silence my LSP
        assert proj_scr_2norm is not None  # to silence my LSP

        tr_mean = np.mean(temp_rp_2norm, axis=0)
        tr_sem = np.std(temp_rp_2norm, ddof=1, axis=0) / np.sqrt(realizations)
        ps_mean = np.mean(proj_scr_2norm, axis=0)
        # FIXME: This statistic does not account for our finite sample of the scrooge ensemble
        ps_sem = np.std(proj_scr_2norm, ddof=1, axis=0) / np.sqrt(realizations)

        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, tr_mean[:,i], label=rf'$k={k}$', color=f'C{i}')
            ax.fill_between(tau_series, tr_mean[:,i]-tr_sem[:,i], tr_mean[:,i]+tr_sem[:,i], alpha=0.5, color=f'C{i}')
        m1 = tau_series[1:]**-1 > 10**-10
        m2 = tau_series[1:]**(-1/2) > 10**-10
        ax.plot(tau_series[1:][m1], (tau_series[1:]**-1)[m1], label=r'$\tau^{-1}$', linestyle='dashed', color=f'C{k_series.size}')
        ax.plot(tau_series[1:][m2], (tau_series[1:]**(-1/2))[m2], label=r'$\tau^{-1/2}$', linestyle='dashed', color=f'C{k_series.size+1}')
        ax.set(
            xlabel=r'$\tau$',
            ylabel=r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$',
            xscale='log',
            yscale='log'
        )
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.suptitle(p_str, y=1.1)
        save_name = os.path.join(ut.FIG_DIR, 'temp_rp_2norm_' + temp_note(realizations, k_series[-1]) + '.png')
        fig.savefig(save_name, **ut.FIG_SAVE_OPTIONS)
        print(f'Temporal and Random Phase Figure saved at: {save_name}')
        
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, ps_mean[:,i], label=rf'$k={k}$', color=f'C{i}')
            ax.fill_between(tau_series, ps_mean[:,i]-ps_sem[:,i], ps_mean[:,i]+ps_sem[:,i], alpha=0.5, color=f'C{i}')
        ax.set(
            xlabel=r'$\tau$',
            ylabel=r'$||\rho_\text{Proj.}^{(k)} - \sum_{z=1}^{D_B} \langle z |\sigma_B|z\rangle \rho_\text{Scr}^{(k)}(\hat{\sigma}_{A|z})||_2$',
            xscale='log',
            yscale='log'
        )
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.suptitle(p_str, y=1.1)
        save_name = os.path.join(ut.FIG_DIR, 'proj_scr_2norm_' + proj_scr_note(realizations, k_series[-1]) + '.png')
        fig.savefig(save_name, **ut.FIG_SAVE_OPTIONS)
        print(f'Projected and Scrooge Figure saved at: {save_name}')
