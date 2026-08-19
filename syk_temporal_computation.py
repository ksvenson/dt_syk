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
import utilities as ut


if __name__ == '__main__':
    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    dm.config.L = 6

    # SYK Parameters
    N = 2 * dm.config.L
    q = 4
    g = 1
    realizations = 100
    H_str = 'SYK42'

    # Experiment Parameters
    seed = 628**3  # the better circle constant!
    psi0_str = 'mark37'
    theta = 0.6
    psi0 = comp.evolve(op.index_sum(op.sigmaz()), st.State(0), -theta / 2)
    psi0_np = psi0.to_numpy(to_all=True)
    k_series = np.arange(2) + 1
    log_tmax = 8
    ns = 100
    tau_series = np.logspace(0, log_tmax, num=ns)
    nbins = 100
    hist_cutoff = 0.1
    if H_str == 'SYK':
        exp_note = lambda task_num, k: f'{H_str}_N{N}_q{q}_seed{seed}_num{task_num}_psi0{psi0_str}_k{k}_logtmax{log_tmax}_ns{ns}'
        H_note = lambda task_num: f'N{N}_q{q}_seed{seed}_num{task_num}'
    elif H_str == 'SYK42':
        exp_note = lambda task_num, k: f'{H_str}_N{N}_g{g:.4f}_seed{seed}_num{task_num}_psi0{psi0_str}_k{k}_logtmax{log_tmax}_ns{ns}'
        H_note = lambda task_num: f'N{N}_g{g:.4f}_seed{seed}_num{task_num}'
    else:
        raise NotImplementedError(f'Unsupported SYK Hamiltonian: {H_str}')
    
    # Hamiltonian construction
    # Done on one rank so that rng is handled safely.
    if rank == 0:
        rng = np.random.default_rng(seed)
        for task_num in range(realizations):
            print(f'Constructing H {task_num}')
            # cache the result, distribute to other ranks later
            if H_str == 'SYK':
                _ = hf.SYK_numpy(q, rng, note=H_note(task_num))
            elif H_str == 'SYK42':
                _ = hf.SYK42_numpy(g, rng, note=H_note(task_num))
            else:
                raise NotImplementedError(f'Unsupported SYK Hamiltonian: {H_str}')
    comm.Barrier()

    # Task allocation across ranks
    task_alloc = [np.arange(rank, realizations, size) for rank in range(size)]
    task_counts = np.array([tasks.size for tasks in task_alloc])
    cum_task_counts = np.zeros(size, dtype=np.int32)
    cum_task_counts[1:] = np.cumsum(task_counts[:-1])

    # "tr" short for (temporal - random phase) 2-norm
    local_tr = np.empty((task_counts[rank], k_series.size, tau_series.size), dtype=np.float64)
    
    # Collecting all energy differences into bins
    bin_edges = np.linspace(0, hist_cutoff, num=nbins+1)
    local_diff_counts = np.zeros((k_series.size, nbins), dtype=np.int32)

    # Collecting the minimum energy differences
    local_min_diffs = np.empty((task_counts[rank], k_series.size))

    # Computations
    for i, task_num in enumerate(task_alloc[rank]):
        print(f'Rank {rank}: {i+1} / {task_counts[rank]}')
        if H_str == 'SYK':
            H_np = hf.SYK_numpy(None, None, note=H_note(task_num), require_cache=True)
        elif H_str == 'SYK42':
            H_np = hf.SYK42_numpy(None, None, note=H_note(task_num), require_cache=True)
        else:
            raise NotImplementedError(f'Unsupported SYK Hamiltonian: {H_str}')
        eigs = la.eig_system(H_np, note=H_str+'_'+H_note(task_num))
        pops = la.get_pops(eigs['evecs'], psi0_np)

        for k_idx, k in enumerate(k_series):
            tools.mpi_print(f'k={k}: Computing temp_norm_squared_exact')
            local_tr[i, k_idx, :] = tf.temp_square_2norm_exact(
                eigs['evals'],
                pops,
                k,
                tau_series,
                verbose=rank==0,
                note=exp_note(task_num, k)
            )
            tools.mpi_print(f'f={k}: Computing eval diffs')
            diff_output = tf.eng_diffs(
                eigs['evals'],
                k,
                bin_edges,
                verbose=False,
                note=f'{H_str}_{H_note(task_num)}_k{k}_nbins{nbins}_cf{hist_cutoff}'
            )
            local_diff_counts[k_idx, :] += diff_output['counts']
            local_min_diffs[i, k_idx] = diff_output['min']

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
    
    diff_counts = np.empty_like(local_diff_counts)
    comm.Allreduce(local_diff_counts, diff_counts)

    min_diffs = None
    recv_buf = None
    if rank == 0:
        min_diffs = np.empty((realizations, k_series.size))
        recv_buf = [
            min_diffs,
            (
                k_series.size * task_counts,
                k_series.size * cum_task_counts
            )
        ]
    comm.Gatherv(local_min_diffs, recv_buf, root=0)

    # Final statistics and plotting
    if rank == 0:
        # TODO: add support for SYK42 Hamiltonian
        if H_str == 'SYK':
            p_str = '\n'.join((
                'Parameters:',
                rf'Hamiltonian: {H_str}, $N = {N}$, $q = {q}$',
                f'Disorder Realizations: {realizations}',
                f'Seed: {seed}',
                rf'$|\psi_0\rangle$: {psi0_str}',
            ))
        elif H_str == 'SYK42':
            p_str = '\n'.join((
                'Parameters:',
                rf'Hamiltonian: {H_str}, $N = {N}$, $g = {g}$',
                f'Disorder Realizations: {realizations}',
                f'Seed: {seed}',
                rf'$|\psi_0\rangle$: {psi0_str}',
            ))
        else:
            raise NotImplementedError(f'Unsupported SYK Hamiltonian: {H_str}')

        # Temporal and Random Phase Comparison
        temp_rp_2norm = np.sqrt(temp_rp_2norm)  # take square root for 2-norm
        mean = np.mean(temp_rp_2norm, axis=0)
        sem = np.std(temp_rp_2norm, ddof=1, axis=0) / np.sqrt(temp_rp_2norm.shape[0])
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, mean[i], label=rf'$k={k}$', color=f'C{i}')
            ax.fill_between(tau_series, mean[i]-sem[i], mean[i]+sem[i], alpha=0.5, color=f'C{i}')
        m1 = tau_series**-1 > 10**-10
        m2 = tau_series**(-1/2) > 10**-10
        ax.plot(tau_series[m1], (tau_series**-1)[m1], label=r'$\tau^{-1}$', linestyle='dashed', color=f'C{k_series.size}')
        ax.plot(tau_series[m2], (tau_series**(-1/2))[m2], label=r'$\tau^{-1/2}$', linestyle='dashed', color=f'C{k_series.size+1}')
        ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
        
        ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.suptitle(p_str, y=1.1)
        fig.savefig(os.path.join(ut.FIG_DIR, 'temp_rp_2norm_' + exp_note('X', 'X') + f'_dr{realizations}.svg'), **ut.FIG_SAVE_OPTIONS)

        # Energy Difference Histogram
        fig, ax = plt.subplots(1, k_series.size, figsize=ut.fig_size(k_series.size, 1))
        diff_counts = diff_counts / np.sum(diff_counts, axis=-1, keepdims=True)
        for i, k in enumerate(k_series):
            ax[i].stairs(diff_counts[i], bin_edges)
            ax[i].set(
                xlabel=r'$\sum_{i=1}^k E_{\alpha_i} - E_{\beta_i}$',
                ylabel='Counts',
                yscale='log',
                title=rf'$k={k}$' + '\n' + f'min: {np.min(min_diffs[:, i])}'
            )
        fig.suptitle(p_str, y=1.1)
        fig.savefig(os.path.join(ut.FIG_DIR, 'eng_diffs_' + exp_note('X', 'X') + f'_dr{realizations}.svg'), **ut.FIG_SAVE_OPTIONS)

        # Minimum Energy Difference Histogram
        fig, ax = plt.subplots(1, k_series.size, figsize=ut.fig_size(k_series.size, 1))
        for i, k in enumerate(k_series):
            ax[i].hist(min_diffs[:, i], histtype='step')
            ax[i].set(
                xlabel=r'$\min \sum_{i=1}^k E_{\alpha_i} - E_{\beta_i}$',
                ylabel='Counts',
                title=rf'$k={k}$' + '\n' + f'min: {np.min(min_diffs[:, i])}'
            )
        fig.suptitle(p_str, y=1.1)
        fig.savefig(os.path.join(ut.FIG_DIR, 'min_eng_diffs_' + exp_note('X', 'X') + f'_dr{realizations}.svg'), **ut.FIG_SAVE_OPTIONS)

