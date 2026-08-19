import dynamite as dm
import dynamite.computations as comp
import dynamite.operators as op
import dynamite.states as st
import dynamite.tools as tools
import numpy as np
import matplotlib.pyplot as plt
import os

import linalg as la
import hamiltonian_factory as hf
import temporal_funcs as tf
import utilities as ut


if __name__ == '__main__':
    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()

    # Experiment Parameters
    dm.config.L = 6
    psi0_str = 'mark37'
    theta = 0.6
    psi0 = comp.evolve(op.index_sum(op.sigmaz()), st.State(0), -theta / 2)
    psi0_np = psi0.to_numpy()
    k_series = np.arange(3) + 1
    log_tmax = 8
    ns = 100
    tau_series = np.logspace(0, log_tmax, num=ns)
    nbins = 100
    hist_cutoff = 0.1

    # MFIM Parameters
    H_str = 'MFIM'
    hx=(np.sqrt(5) + 5) / 8
    hz=(np.sqrt(5) + 1) / 4
    H = hf.MFIM(hx=hx, hz=hz)
    base_note = f'L{dm.config.L}_{H_str}_hx{hx:.4f}_hz{hz:.4f}'

    # Computations
    evals = np.empty(2**dm.config.L)
    pops = np.empty(2**dm.config.L)
    diff_counts = np.empty((k_series.size, nbins), dtype=np.int32)
    bin_edges = np.linspace(0, hist_cutoff, num=nbins+1)
    min_diff = np.empty(k_series.shape)
    if rank == 0:
        tools.mpi_print('Computing eigs')
        eigs = la.eig_system(
            H.to_numpy(sparse=False),
            note=base_note
        )
        evals = eigs['evals']
        pops = la.get_pops(eigs['evecs'], psi0_np)
    comm.Bcast(evals, root=0)
    comm.Bcast(pops, root=0)

    tools.mpi_print('Computing Eigenvalue Differences')
    for i, k in enumerate(k_series):
        output = tf.eng_diffs(
            evals,
            k,
            bin_edges,
            comm=comm,
            verbose=(rank==0),
            note=f'{base_note}_k{k}_nbins{nbins}_cf{hist_cutoff}'
        )
        diff_counts[i] = output['counts']
        min_diff[i] = output['min']
    
    tools.mpi_print('Computing temp_norm_squared_exact')
    temp_2norm = np.zeros((tau_series.size, k_series.size))
    for k_idx, k in enumerate(k_series):
        temp_2norm[:, k_idx] = tf.temp_square_2norm_exact(
            evals,
            pops,
            k,
            tau_series,
            comm=comm,
            verbose=(rank==0),
            note=f'{base_note}_psi0{psi0_str}_k{k}_tmax{log_tmax}_ns{ns}'
        )
    diff_exact = np.sqrt(temp_2norm)  # we already subtracted the RPE term in this computation

    # Plotting
    if rank == 0:
        p_str = '\n'.join((
            'Parameters:',
            rf'Hamiltonian: {H_str}, $L={dm.config.L}$, $hx = {hx:.4f}$, $hz = {hz:.4f}$',
            rf'$|\psi_0\rangle$: {psi0_str}',
        ))

        # 2-Norm
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, diff_exact[:, i], label=rf'$k={k}$')
        ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
        
        ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.suptitle(p_str, y=1.05)
        fig.savefig(os.path.join(ut.FIG_DIR, f'temp_rp_2norm_{base_note}_psi0{psi0_str}.svg'), **ut.FIG_SAVE_OPTIONS)

        # Energy Difference Histogram
        fig, ax = plt.subplots(1, k_series.size, figsize=ut.fig_size(k_series.size, 1))
        density = diff_counts / np.sum(diff_counts, axis=-1, keepdims=True)
        for i, k in enumerate(k_series):
            ax[i].stairs(density[i], bin_edges)
            ax[i].set(
                xlabel=r'$\sum_{i=1}^k E_{\alpha_i} - E_{\beta_i}$',
                ylabel='Counts',
                title='\n'.join((
                    rf'$k={k}$',
                    f'Total below {hist_cutoff}: {np.sum(diff_counts[i]):.3e}',
                    f'min: {min_diff[i]:.3e}'
                ))
            )
        fig.suptitle(p_str, y=1.1)
        fig.savefig(os.path.join(ut.FIG_DIR, f'eng_diffs_{base_note}_psi0{psi0_str}.svg'), **ut.FIG_SAVE_OPTIONS)

