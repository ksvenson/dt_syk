import os
import numpy as np
import matplotlib.pyplot as plt

import utilities as ut

TACC_CACHE = './tacc_cache'


if __name__ == '__main__':
    # Sweep Parameters
    N_series = np.arange(16, 24+2, 2)
    g_series = np.array([0, 10, 100], dtype=float)
    # N_series = np.arange(16, 18+2, 2)
    # g_series = np.array([0, 10], dtype=float)

    # Held constant during sweep
    k_series = np.arange(3) + 1
    DA = 4
    n_scr = 1000
    psi0_str = '0'
    realizations = 128
    base_seed = 628**3
    dt = 1.0
    log_tmax = 5
    tau_series = dt * np.arange(0, int(10**log_tmax / dt))
    
    # Cache keys
    H_title = 'SYK42'
    H_str = lambda N, g, r: f'{H_title}_N{N}_g{g}_base-seed{base_seed}_r{r}'
    temp_note = lambda N, g, r, k: f'{H_str(N, g, r)}_psi0{psi0_str}_k{k}_dt{dt}_logtmax{log_tmax}'
    proj_note = lambda N, g, r, k: f'{temp_note(N, g, r, k)}_DA{DA}'
    scr_note = lambda N, g, r, k: f'{H_str(N, g, r)}_psi0{psi0_str}_k{k}_nscr{n_scr}_DA{DA}'
    proj_scr_note = lambda N, g, r, k: f'{proj_note(N, g, r, k)}_nscr{n_scr}'

    p_str = '\n'.join((
        'Parameters:',
        rf'Hamiltonian: {H_title}',
        f'Disorder Realizations: {realizations}',
        f'Scrooge Samples: {n_scr}',
        f'Base Seed: {base_seed}',
        rf'$|\psi_0\rangle$: {psi0_str}',
        f'dt: {dt}',
        f'DA: {DA}'
    ))
    
    # Initializing arrays
    temp_rp_2norm = np.empty((
        N_series.size,
        g_series.size,
        realizations,
        tau_series.size,
        k_series.size
    ))
    proj_scr_2norm = np.empty_like(temp_rp_2norm)
    
    # Loading data
    for N_idx, N in enumerate(N_series):
        for g_idx, g in enumerate(g_series):
            try:
                print(f'Attempting load ({N}, {g})')
                temp_rp_2norm[N_idx, g_idx] = np.load(os.path.join(TACC_CACHE, 'temp_rp_2norm_' + temp_note(N, g, realizations, k_series[-1]) + '.npy'))
                proj_scr_2norm[N_idx, g_idx] = np.load(os.path.join(TACC_CACHE, 'proj_scr_2norm_' + proj_scr_note(N, g, realizations, k_series[-1]) + '.npy'))
            except FileNotFoundError:
                print(f'WARNING: Cache for configuration (N, g) = ({N}, {g}) not found. Omitting...')
                temp_rp_2norm[N_idx, g_idx] = np.full(temp_rp_2norm.shape[2:], np.nan)

    # Statistics
    tr_mean = np.mean(temp_rp_2norm, axis=2)
    tr_sem = np.std(temp_rp_2norm, ddof=1, axis=2) / np.sqrt(realizations)
    ps_mean = np.mean(proj_scr_2norm, axis=2)
    ps_sem = np.std(proj_scr_2norm, ddof=1, axis=2) / np.sqrt(realizations)
    
    # Plotting
    tr_fig, tr_ax = plt.subplots(
        nrows=k_series.size,
        ncols=g_series.size,
        sharey=True,
        figsize=ut.fig_size(g_series.size, k_series.size)
    )
    ps_fig, ps_ax = plt.subplots(
        nrows=k_series.size,
        ncols=g_series.size,
        sharey=True,
        figsize=ut.fig_size(g_series.size, k_series.size)
    )
    cp_fig, cp_ax = plt.subplots(
        nrows=k_series.size,
        ncols=g_series.size,
        sharey=True,
        figsize=ut.fig_size(g_series.size, k_series.size)
    )
    means = (tr_mean, ps_mean)
    sems = (tr_sem, ps_sem)
    axes = (tr_ax, ps_ax, cp_ax)
    figs = (tr_fig, ps_fig, cp_fig)
    tr_ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RP.}^{(k)}||_2$'
    ps_ylabel = r'$||\rho_\text{Proj.}^{(k)} - \sum_{z=1}^{D_B} \langle z |\sigma_B|z\rangle \rho_\text{Scr}^{(k)}(\hat{\sigma}_{A|z})||_2$'
    for k_idx, k in enumerate(k_series):
        for g_idx, g in enumerate(g_series):
            for N_idx, N in enumerate(N_series):
                m = (N_idx, g_idx, slice(None), k_idx)
                for i in range(2):
                    # Plotting means
                    axes[i][k_idx, g_idx].plot(tau_series, means[i][m], label=rf'$N={N}$', color=f'C{N_idx}')
                    # Plotting Errors
                    axes[i][k_idx, g_idx].fill_between(
                        tau_series,
                        means[i][m] - sems[i][m],
                        means[i][m] + sems[i][m],
                        alpha=0.5,
                        color=f'C{N_idx}'
                    )
                cp_ax[k_idx, g_idx].plot(tr_mean[m], ps_mean[m], label=rf'$N={N}$', color=f'C{N_idx}')
            # Labels that go on every tile
            for i in range(3):
                axes[i][k_idx, g_idx].set(
                    xscale='log',
                    yscale='log'
                )
                axes[i][k_idx, g_idx].legend(**ut.LEGEND_OPTIONS)
            tr_ax[k_idx, g_idx].set(xlabel=r'$\tau$')
            ps_ax[k_idx, g_idx].set(xlabel=r'$\tau$')
            cp_ax[k_idx, g_idx].set(xlabel=tr_ylabel)
            # Column labels
            pad = 5
            if k_idx == 0:
                for i in range(3):
                    axes[i][0, g_idx].annotate(
                        rf'$g={g}$',
                        xy=(0.5, 1),
                        xytext=(0, pad),
                        xycoords='axes fraction',
                        textcoords='offset points',
                        size='large',
                        ha='center',
                        va='baseline'
                    )
            # Row labels
            if g_idx == 0:
                tr_ax[k_idx, 0].set(ylabel=tr_ylabel)
                ps_ax[k_idx, 0].set(ylabel=ps_ylabel)
                cp_ax[k_idx, 0].set(ylabel=ps_ylabel)
                for i in range(3):
                    axes[i][k_idx, 0].annotate(
                        rf'$k={k}$',
                        xy=(0, 0.5),
                        xytext=(-axes[i][k_idx, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[i][k_idx, 0].yaxis.label,
                        textcoords='offset points',
                        size='large',
                        ha='right',
                        va='center'
                    )
    for i in range(3):
        figs[i].suptitle(p_str, y=1.05)

    tr_fig.savefig(
        os.path.join(
            ut.FIG_DIR,
            'N_g_sweep_temp_rp_2norm_' + temp_note(N_series[-1], g_series[-1], realizations, k_series[-1]) + '.png'
        ),
        **ut.FIG_SAVE_OPTIONS
    )
    ps_fig.savefig(
        os.path.join(
            ut.FIG_DIR,
            'N_g_sweep_proj_scr_2norm_' + proj_scr_note(N_series[-1], g_series[-1], realizations, k_series[-1]) + '.png'
        ),
        **ut.FIG_SAVE_OPTIONS
    )
    cp_fig.savefig(
        os.path.join(
            ut.FIG_DIR,
            'N_g_sweep_compare_designs' + proj_scr_note(N_series[-1], g_series[-1], realizations, k_series[-1]) + '.png'
        ),
        **ut.FIG_SAVE_OPTIONS
    )


