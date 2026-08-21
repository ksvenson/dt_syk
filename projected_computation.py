import os

import numpy as np
import matplotlib.pyplot as plt

import dynamite as dm
import dynamite.tools as tools
import dynamite.states as st

import utilities as ut
import hamiltonian_factory as hf
import projected_funcs as pf


if __name__ == '__main__':
    # MPI config
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # SYK Parameters
    N = 20   # total number of fermions
    NA = 2   # number of fermions in subsystem
    g = 1    # coupling to 2-body interactions
    dr = 10  # dr stands for disorder realizations

    # RNG config
    base_seed = 628**3  # the better circle constant!
    rng = lambda r: np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(r,)))
    H_title = 'SYK42'
    H_str = lambda r: f'{H_title}_N{N}_g{g}_base-seed{base_seed}_r{r}'
    H_func = hf.SYK42
    H_args = lambda r: [g, rng(r)]

    # dynamite config
    dm.config.L = N // 2
    D = 2**dm.config.L
    DA = 2**(NA // 2)

    # Experiment Parameters
    psi0_str = '0'
    psi0 = st.State(0)
    k_series = np.arange(3) + 1
    log_tmax = 3
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
    
    # Computations
    proj_norm = np.empty((2, dr,) + k_series.shape + tau_series.shape)
    for r in range(dr):  # TODO: make this loop into a slurm job array
        print(f'Realization {r+1}/{dr}')
        H = H_func(*(H_args(r)))
        pm = pf.proj_moment(H, psi0, tau_series, k_series, DA, note=exp_note(r, f'{k_series[0]}-{k_series[-1]}'))
        # TODO: here we would insert the computation of the scrooge ensemble
        for k_idx, k in enumerate(k_series):
            for p in range(2):
                proj_norm[p, r, k_idx] = pf.norm(pm[str(k)], p+1, axis=(1, 2))
    proj_norm_mean = np.mean(proj_norm, axis=1)
    proj_norm_sem = np.std(proj_norm, ddof=1, axis=1) / np.sqrt(dr)
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=ut.fig_size(2, 1))
    for p in range(2):
        for k_idx, k in enumerate(k_series):
            ax[p].plot(tau_series, proj_norm_mean[p, k_idx], label=rf'$k={k}$')
            ax[p].fill_between(
                tau_series,
                proj_norm_mean[p, k_idx] - proj_norm_sem[p, k_idx],
                proj_norm_mean[p, k_idx] + proj_norm_sem[p, k_idx],
                alpha=0.5
            )
        ax[p].set(
            xlabel=r'$\tau$',
            ylabel=r'$||\rho_\text{Proj.}^{(k)}||_' + f'{p+1}$',
            xscale='log',
            title=rf'{p+1}-norm'
        )
        ax[p].legend(**ut.LEGEND_OPTIONS)
    fig.suptitle(p_str, y=1.1)
    fig.savefig(os.path.join(ut.FIG_DIR, f'proj_norm_{exp_note(dr, f"{k_series[0]}-{k_series[-1]}")}.svg'), **ut.FIG_SAVE_OPTIONS)

