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

    # SYK Parameters
    N = 12
    q = 4
    H_str = 'SYK'
    realizations = 100

    # Experiment Parameters
    seed = 628**3  # the better circle constant!
    dm.config.L = N // 2
    psi0_str = 'mark37'
    theta = 0.6
    psi0 = comp.evolve(op.index_sum(op.sigmaz()), st.State(0), -theta / 2)
    psi0_np = psi0.to_numpy(to_all=True)
    k_series = np.arange(2) + 1
    log_tmax = 8
    ns = 100
    tau_series = np.logspace(0, log_tmax, num=ns)
    exp_note = lambda task_num, k: f'{H_str}_N{N}_q{q}_seed{seed}_num{task_num}_psi0{psi0_str}_k{k}_logtmax{log_tmax}_ns{ns}'
    H_note = lambda task_num: f'N{N}_q{q}_seed{seed}_num{task_num}'
    
    # Hamiltonian construction
    # Done on one rank so that rng is handled safely.
    if rank == 0:
        rng = np.random.default_rng(seed)
        for task_num in range(realizations):
            print(f'Constructing H {task_num}')
            # cache the result, distribute to other ranks later
            _ = hf.SYK_numpy(q, rng, note=H_note(task_num))
    comm.Barrier()

    # Task allocation across ranks
    task_alloc = [np.arange(rank, realizations, size) for rank in range(size)]
    counts = np.array([tasks.size for tasks in task_alloc])
    local = np.empty((counts[rank], k_series.size, tau_series.size), dtype=np.float64)
    local_sub_size = np.prod(local.shape[1:])
    displacements = np.zeros(size, dtype=np.int32)
    displacements[1:] = local_sub_size * np.cumsum(counts[:-1])

    # Computations
    for i, task_num in enumerate(task_alloc[rank]):
        print(f'Rank {rank}: {i+1} / {counts[rank]}')
        H_np = hf.SYK_numpy(None, None, note=H_note(task_num), require_cache=True)
        eigs = la.eig_system(H_np, note=H_str+'_'+H_note(task_num))
        pops = la.get_pops(eigs['evecs'], psi0_np)
        
        for k_idx, k in enumerate(k_series):
            local[i, k_idx, :] = tf.temp_square_2norm_exact(
                eigs['evals'],
                pops,
                k,
                tau_series,
                verbose=False,
                note=exp_note(task_num, k)
            )

    # Gathering rank computations
    temp_rp_2norm = None
    recv_buf = None
    if rank == 0:
        temp_rp_2norm = np.empty((realizations, *local.shape[1:]), dtype=np.float64)
        recv_buf = [
            temp_rp_2norm,
            (counts * local_sub_size, displacements)
        ]
    comm.Gatherv(local, recv_buf, root=0)

    # Final statistics and plotting
    if rank == 0:
        assert temp_rp_2norm is not None
        temp_rp_2norm = np.sqrt(temp_rp_2norm)
        mean = np.mean(temp_rp_2norm, axis=0)
        sem = np.std(temp_rp_2norm, ddof=1, axis=0) / np.sqrt(temp_rp_2norm.shape[0])
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, mean[i], label=rf'$k={k}$', color=f'C{i}')
            ax.fill_between(tau_series, mean[i]-sem[i], mean[i]+sem[i], alpha=0.5, color=f'C{i}')
        m1 = tau_series**-1 > 10**-2
        m2 = tau_series**(-1/2) > 10**-2
        ax.plot(tau_series[m1], (tau_series**-1)[m1], label=r'$\tau^{-1}$', linestyle='dashed', color=f'C{k_series.size}')
        ax.plot(tau_series[m2], (tau_series**(-1/2))[m2], label=r'$\tau^{-1/2}$', linestyle='dashed', color=f'C{k_series.size+1}')
        ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
        
        ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
        ax.legend(**ut.LEGEND_OPTIONS)

        p_str = '\n    '.join((
            'Parameters:',
            rf'Hamiltonian: {H_str}, $N = {N}$, $q = {q}$',
            f'Disorder Realizations: {realizations}',
            f'Seed: {seed}',
            rf'$|\psi_0\rangle$: {psi0_str}',
        ))
        ax.text(
            1.05, 0.5,
            p_str,
            transform=ax.transAxes,
            verticalalignment='center'
        )

        fig.savefig(os.path.join(ut.FIG_DIR, 'temp_rp_2norm_' + exp_note('X', 'X') + f'_dr{realizations}.svg'), **ut.FIG_SAVE_OPTIONS)

