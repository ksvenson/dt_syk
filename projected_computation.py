import os

import numpy as np
import matplotlib.pyplot as plt

import dynamite as dm
import dynamite.tools as tools
import dynamite.states as st
import dynamite.computations as comp

import utilities as ut
import hamiltonian_factory as hf


if __name__ == '__main__':
    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # SYK Parameters
    N = 12   # total number of fermions
    NA = 2   # number of fermions in subsystem
    g = 1    # coupling to 2-body interactions
    dr = 10  # dr stands for disorder realizations

    # RNG config
    base_seed = 628**3  # the better circle constant!
    rng = [np.random.SeedSequence(base_seed, spawn_key=(r,)) for r in range(dr)]
    H_title = 'SYK42'
    H_str = lambda r: f'{H_title}_N{N}_g{g}_base-seed{base_seed}_r{r}'
    H_func = hf.SYK42
    H_args = lambda r: [g, rng[r]]

    # Experiment Parameters
    psi0_str = '0'
    psi0 = st.State(0)
    k_series = np.arange(2) + 1
    log_tmax = 8
    ns = 100
    tau_series = np.logspace(0, log_tmax, num=ns)
    exp_note = lambda r, k: f'{H_str(r)}_psi0{psi0_str}_k{k}_logtmax{log_tmax}_ns{ns}'
    p_str = '\n'.join((
        'Parameters:',
        rf'Hamiltonian: {H_str(dr)}',
        f'Disorder Realizations: {dr}',
        f'Base Seed: {base_seed}',
        rf'$|\psi_0\rangle$: {psi0_str}',
    ))

    # dynamite config
    dm.config.L = N // 2
    D = 2**dm.config.L
    DA = 2**(NA // 2)

    proj_1norm = np.empty((dr,) + k_series.shape + tau_series.shape)
    proj_2norm = np.empty_like(proj_1norm)
    for r in range(dr):  # TODO: make this loop into a slurm job array
        H = H_func(*(H_args(r)))
        psit = np.empty((tau_series.size, D))
        for i, tau in enumerate(tau_series):
            psit[i, :] = comp.evolve(H, psi0, tau).to_numpy(to_all=True)
        psit = psit.reshape(tau_series.size, -1, DA)
        for k_idx, k in enumerate(k_series):
            # TODO: post-select on parity
            # TODO: proper normalization
            moment = np.einsum('abc,abd->acd', psit, psit.conj(), optimize=True)

            # uncomment if we need to chunk:
            # chunk = 256
            # for c in range(0, psit.shape[1], chunk):  # chunk over system B size
            #     moment = np.einsum('abc,abd->acd', psit[:, c:c+chunk, :], psit[:, c:c+chunk, :].conj())

            # TODO: here we would insert the computation of the scrooge ensemble
            proj_1norm[r, k_idx] = np.linalg.norm(moment, ord='nuc', axis=(1, 2))
            proj_2norm[r, k_idx] = np.linalg.norm(moment, ord='fro', axis=(1, 2))
    proj_1norm = np.mean(proj_1norm, axis=0)
    proj_2norm = np.mean(proj_2norm, axis=0)
    proj_1norm_sem = np.std(proj_1norm, ddof=1, axis=0) / np.sqrt(dr)
    proj_2norm_sem = np.std(proj_2norm, ddof=1, axis=0) / np.sqrt(dr)

    fig, ax = plt.subplots(1, 2, figsize=ut.fig_size(2, 1))
    for k_idx, k in enumerate(k_series):
        ax[0].plot(tau_series, proj_1norm[k_idx], label=rf'$k={k}$')
        ax[1].plot(tau_series, proj_2norm[k_idx], label=rf'$k={k}$')
        ax[0].fill_between(
            tau_series,
            proj_1norm[k_idx] - proj_1norm_sem[k_idx],
            proj_1norm[k_idx] + proj_1norm_sem[k_idx],
            alpha=0.5
        )
        ax[1].fill_between(
            tau_series,
            proj_2norm[k_idx] - proj_2norm_sem[k_idx],
            proj_2norm[k_idx] + proj_2norm_sem[k_idx],
            alpha=0.5
        )
    ax[0].set(
        xlabel=r'$\tau$',
        ylabel=r'$||\rho_\text{Proj.}^{(k)}||_1',
        title=rf'1-norm'
    )
    ax[1].set(
        xlabel=r'$\tau$',
        ylabel=r'$||\rho_\text{Proj.}^{(k)}||_2',
        title=rf'2-norm'
    )
    ax[0].legend(**ut.LEGEND_OPTIONS)
    ax[1].legend(**ut.LEGEND_OPTIONS)
    fig.suptitle(p_str, y=1.1)
    fig.savefig(os.path.join(ut.FIG_DIR, f'proj_norm_{exp_note(dr, k_series[-1])}'))

