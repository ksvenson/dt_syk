import os
import numpy as np
import matplotlib.pyplot as plt

import utilities as ut

TACC_CACHE = './tacc_computation_cache'


if __name__ == '__main__':
    # Sweep Parameters
    N_series = np.array([18, 20])
    g_series = np.array([0, 100, 1000])

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
                temp_rp_2norm[N_idx, g_idx] = np.load(os.path.join(TACC_CACHE, 'temp_rp_2norm' + temp_note(N, g, realizations, k_series[-1]) + '.npy'))
                proj_scr_2norm[N_idx, g_idx] = np.load(os.path.join(TACC_CACHE, 'proj_scr_2norm' + proj_scr_note(N, g, realizations, k_series[-1]) + '.npy'))
            except FileNotFoundError:
                print(f'WARNING: Cache for configuration (N, g) = ({N}, {g}) not found. Omitting...')
                temp_rp_2norm[N_idx, g_idx] = np.full(temp_rp_2norm.shape[2:], np.nan)

    # Statistics
    tr_mean = np.mean(temp_rp_2norm, axis=2)
    tr_sem = np.std(temp_rp_2norm, ddof=1, axis=0) / np.sqrt(realizations)
    ps_mean = np.mean(proj_scr_2norm, axis=2)
    ps_sem = np.std(proj_scr_2norm, ddof=1, axis=0) / np.sqrt(realizations)
    
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=k_series.size, sharey=True, figsize=ut.fig_size(k_series.size, 1))
    for N_idx, N in enumerate(N_series):
        for g_idx, g in enumerate(g_series):
            pass



