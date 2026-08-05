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
    dm.config.L = 8
    psi0_str = 'mark37'
    theta = 0.6
    psi0 = comp.evolve(op.index_sum(op.sigmaz()), st.State(0), -theta / 2)
    psi0_np = psi0.to_numpy()
    k_series = np.arange(2) + 1
    log_tmax = 8
    ns = 100
    tau_series = np.logspace(0, log_tmax, num=ns)
    num_bins = 100
    cutoff = None

    # MFIM Parameters
    H_str = 'MFIM'
    hx=(np.sqrt(5) + 5) / 8
    hz=(np.sqrt(5) + 1) / 4
    H = hf.MFIM(hx=hx, hz=hz)
    base_note = f'L{dm.config.L}_{H_str}_hx{hx:.4f}_hz{hz:.4f}'

    # Computations
    evals = np.empty(2**dm.config.L)
    pops = np.empty(2**dm.config.L)
    diff_counts = np.empty((k_series.size, num_bins))
    bin_edges = np.empty((k_series.size, num_bins+1))
    if rank == 0:
        tools.mpi_print('Computing eigs')
        eigs = la.eig_system(
            H.to_numpy(sparse=False),
            note=base_note
        )
        evals = eigs['evals']
        pops = la.get_pops(eigs['evecs'], psi0_np)

        tools.mpi_print('Computing Eigenvalue Differences')
        for i, k in enumerate(k_series):
            output = tf.eng_diffs(
                evals,
                k,
                num_bins=num_bins,
                cutoff=cutoff,
                note=f'{base_note}_nbins{num_bins}_cf{cutoff}'
            )
            diff_counts[i] = output['counts']
            bin_edges[i] = output['bin_edges']
    
    comm.Bcast(evals, root=0)
    comm.Bcast(pops, root=0)
    comm.Bcast(diff_counts, root=0)
    comm.Bcast(bin_edges, root=0)
    
    tools.mpi_print('Computing temp_norm_squared_exact')
    temp_2norm = np.zeros((tau_series.size, k_series.size))
    for k_idx, k in enumerate(k_series):
        temp_2norm[:, k_idx] = tf.temp_square_2norm_exact(
            evals,
            pops,
            k,
            tau_series,
            comm=comm,
            note=f'{base_note}_psi0{psi0_str}_k{k}_tmax{log_tmax}_ns{ns}'
        )
    diff_exact = np.sqrt(temp_2norm)  # we already subtracted the RPE term in this computation

    # Plotting
    if rank == 0:
        # 2-Norm
        fig, ax = plt.subplots()
        for i, k in enumerate(k_series):
            ax.plot(tau_series, diff_exact[:, i], label=rf'$k={k}$')
        ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
        
        ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
        ax.legend(**ut.LEGEND_OPTIONS)
        fig.savefig(os.path.join(ut.FIG_DIR, f'temp_rp_2norm_{base_note}_psi0{psi0_str}.svg'), **ut.FIG_SAVE_OPTIONS)

        # Energy Difference Histogram
        fig, ax = plt.subplots(1, k_series.size, layout='constrained')
        density = diff_counts / np.sum(diff_counts, axis=-1, keepdims=True)
        for i, k in enumerate(k_series):
            ax[i].stairs(density[i], bin_edges[i])
            ax[i].set(xlabel=r'$\sum_{i=1}^k E_{\alpha_i} - E_{\beta_i}$', ylabel='Counts', title=rf'$k={k}$')
        fig.savefig(os.path.join(ut.FIG_DIR, f'eng_diffs_{base_note}_psi0{psi0_str}.svg'), **ut.FIG_SAVE_OPTIONS)

