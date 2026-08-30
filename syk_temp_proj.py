import dynamite as dm
import dynamite.computations as comp
import dynamite.operators as op
import dynamite.states as st
import dynamite.tools as tools
import numpy as np
import os
import matplotlib.pyplot as plt

import linalg as la
import hamiltonian_factory as hf
import temporal_funcs as tf
import random_phase as rp
import utilities as ut


if __name__ == '__main__':
    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # SYK Parameters
    N = 12   # total number of fermions
    NA = 2   # number of fermions in subsystem
    g = 100    # coupling to 2-body interactions
    realizations = 10  # disorder realizations

    # RNG config
    base_seed = 628**3  # the better circle constant!
    rng = lambda r: np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(r,)))
    H_title = 'SYK42'
    H_str = lambda r: f'{H_title}_N{N}_g{g}_base-seed{base_seed}_r{r}'
    H_func = hf.SYK42_numpy
    H_args = lambda r: [g, rng(r)]

    # dynamite config
    dm.config.L = N // 2
    D = 2**dm.config.L
    DA = 2**(NA // 2)

    # Experiment Parameters
    k_series = np.arange(3) + 1
    psi0_str = '0'
    psi0 = st.State(0)
    psi0_np = psi0.to_numpy(to_all=True)
    assert psi0_np is not None  # to silence my LSP
    dt = 1.0
    log_tmax = 6
    tau_series = dt * np.arange(0, int(10**log_tmax / dt))
    temp_note = lambda r, k: f'{H_str(r)}_psi0{psi0_str}_k{k}_dt{dt}_logtmax{log_tmax}'
    proj_note = lambda r, k: f'{temp_note(r, k)}_DA{DA}'
    p_str = '\n'.join((
        'Parameters:',
        rf'Hamiltonian: {H_str(realizations)}',
        f'Disorder Realizations: {realizations}',
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
    
    # Computations
    for i, r in enumerate(task_alloc[rank]):  # r stands for realization
        print(f'Rank {rank}: {i+1} / {task_counts[rank]}')
        H_np = H_func(*H_args(r), note=H_str(r))
        eigs = la.eig_system(H_np, note=H_str(r))
        evals = eigs['evals']
        evecs = eigs['evecs']
        psi0_eng_basis = np.einsum('ab,b->a', evecs.conj().T, psi0_np)

        pops = np.abs(psi0_eng_basis)**2
        overlaps = tf.time_evolved_overlaps_and_rhoA(
            evals,
            evecs,
            psi0_eng_basis,
            tau_series,
            k_series,
            DA,
            note=temp_note(r, f'{k_series[0]}-{k_series[-1]}')
        )

        local_tr[i] = tf.temp_square_2norm_sampled(overlaps, k_series)
        for k_idx, k in enumerate(k_series):
            local_tr[i, :, k_idx] = local_tr[i, :, k_idx] - rp.rpe_square_2norm(pops, k, note=temp_note(r, k))
    local_tr = np.sqrt(local_tr)

    # Gathering rank computations
    temp_rp_2norm = None
    recv_buf = None
    if rank == 0:
        temp_rp_2norm = np.empty((realizations, *local_tr.shape[1:]), dtype=np.float64)
        recv_buf = [
            temp_rp_2norm,
            (
                np.prod(local_tr.shape[1:]) * task_counts,     # size of each local array
                np.prod(local_tr.shape[1:]) * cum_task_counts  # displacements
            )
        ]
    comm.Gatherv(local_tr, recv_buf, root=0)
    
    # Final statistics and plotting
    if rank == 0:
        assert temp_rp_2norm is not None  # to silence my LSP
        mean = np.mean(temp_rp_2norm, axis=0)
        sem = np.std(temp_rp_2norm, ddof=1, axis=0) / np.sqrt(temp_rp_2norm.shape[0])
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, mean[:,i], label=rf'$k={k}$', color=f'C{i}')
            ax.fill_between(tau_series, mean[:,i]-sem[:,i], mean[:,i]+sem[:,i], alpha=0.5, color=f'C{i}')
        m1 = tau_series**-1 > 10**-10
        m2 = tau_series**(-1/2) > 10**-10
        ax.plot(tau_series[m1], (tau_series**-1)[m1], label=r'$\tau^{-1}$', linestyle='dashed', color=f'C{k_series.size}')
        ax.plot(tau_series[m2], (tau_series**(-1/2))[m2], label=r'$\tau^{-1/2}$', linestyle='dashed', color=f'C{k_series.size+1}')
        ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
        
        ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.suptitle(p_str, y=1.1)
        fig.savefig(os.path.join(ut.FIG_DIR, 'temp_rp_2norm_' + temp_note(realizations, k_series[-1]) + '.png'), **ut.FIG_SAVE_OPTIONS)

