"""Compare saved slider runs using their actual reporting axes and metadata."""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import thermalization as th


def plot_archives(paths, gap_lines=False, fraction=None, expected_model=None):
    """Plot the four distance curves and the original parametric comparison.

    paths is a nonempty list of production NPZ files, one per (N,g) for SYK or
    one per L for MFIM. A figure never mixes the two models. Runs must
    agree in subsystem, moment orders, random seed, parity, projected-time
    sampling controls, Scrooge sample count, Hamiltonian convention, RP reference,
    energy-grouping tolerance, and MFIM fields/boundary when applicable. Their
    realization counts may differ. Each curve averages distances over disorder;
    bands are disorder SEM, not separate Haar/time-sampling error estimates.

    gap_lines optionally shows saved finite spectral reference medians only on
    the temporal plot. fraction optionally overrides the saved observed threshold.
    The single-Scrooge first moment should vanish mathematically. Its computed
    residual uses a linear vertical axis with automatic upper limits, so discrepancies remain
    visible instead of being hidden by an assumed exact-zero annotation.

    Five figures are written under utilities.FIG_DIR. No states are recomputed.
    Incompatible archives or duplicate (N,g) configurations raise ValueError.
    """
    if not paths:
        raise ValueError('Supply at least one result archive')
    comparisons = [
        ('temp_rp_2norm',r'$||\rho_T^{(k)}-\rho_{RP}^{(k)}||_2$'),
        ('proj_scr_2norm',r'$\mathbb{E}_t||M_k(t)-G_k[\sigma]||_2$'),
        ('proj_single_scr_2norm',r'$\mathbb{E}_t||M_k(t)-S_k[\rho_A(t)]||_2$'),
        ('proj_single_scr_instant_2norm',r'$||M_k(t)-S_k[\rho_A(t)]||_2$')]
    runs, seen, signature = [], set(), None
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            params = json.loads(str(data['metadata']))
            model = params.get('model','SYK')
            if expected_model is not None and model != expected_model:
                raise ValueError(f'Expected {expected_model} archives, found {model}')
            pair = (model,params['L'] if model=='MFIM' else params['N'],params.get('g'))
            if pair in seen:
                raise ValueError(f'Multiple archives for {pair}; choose one')
            seen.add(pair)
            current = (model,)+tuple(params.get(k) for k in ('NA','LA','k_max','n_scr','seed','parity',
                'projected_sampling','projected_dt','time_seed','tmax','hamiltonian_convention',
                'rp_reference','energy_grouping','degeneracy_tolerance',
                'single_scrooge_evaluation','hx','hy','hz','J','boundary','measurement_basis'))
            if signature is not None and signature != current:
                raise ValueError('Archives mix incompatible physical or sampling parameters')
            signature = current
            runs.append((params,data['tau_series'],data['k_series'],
                         {key:data[key] for key,_ in comparisons},
                         data['moment_gap_times'],data['projected_sample_times']))
    model = runs[0][0].get('model','SYK')
    size_key = 'L' if model=='MFIM' else 'N'
    gs = [None] if model=='MFIM' else sorted({p['g'] for p,*_ in runs})
    ks = runs[0][2]
    parametric_index = len(comparisons)
    figures = [plt.subplots(len(ks),len(gs),squeeze=False,
                           figsize=ut.fig_size(len(gs),len(ks))) for _ in range(parametric_index+1)]
    for params,times,orders,values,gaps,sample_times in sorted(runs,key=lambda x:x[0][size_key]):
        col = gs.index(params.get('g'))
        size_label = f'{size_key}={params[size_key]}'
        means = {key:array.mean(axis=0) for key,array in values.items()}
        for row,k in enumerate(orders):
            for index,(key,label) in enumerate(comparisons):
                ax = figures[index][1][row,col]
                x = sample_times.mean(axis=0) if key=='proj_single_scr_instant_2norm' else times
                samples = values[key]
                mean = means[key][:,row]
                sem = samples.std(axis=0,ddof=1)[:,row]/np.sqrt(len(samples)) if len(samples)>1 else np.zeros_like(mean)
                line, = ax.plot(x[1:],mean[1:],label=size_label if model=='MFIM' else f'{size_label}, R={len(samples)}')
                ax.fill_between(x[1:],np.maximum(mean[1:]-sem[1:],0),mean[1:]+sem[1:],alpha=.2)
                th.annotate_timescales(ax,x,mean[:,None],[k],
                    gap_times=gaps[:,row:row+1] if index==0 and (gap_lines or params.get('gap_lines',False)) else None,
                    fraction=fraction if fraction is not None else params.get('thermalization_fraction'),
                    label_prefix=size_label+': ',colors=[line.get_color()])
            figures[parametric_index][1][row,col].plot(means['temp_rp_2norm'][1:,row],means['proj_scr_2norm'][1:,row],label=size_label)
    os.makedirs(ut.FIG_DIR,exist_ok=True)
    for index,(fig,axes) in enumerate(figures):
        for row,k in enumerate(ks):
            for col,g in enumerate(gs):
                ax = axes[row,col]
                numerical_check = index in (2,3) and k==1
                ax.set(xscale='log',yscale='linear' if numerical_check else 'log',title=f'k={k}, MFIM' if model=='MFIM' else f'k={k}, g={g}',
                       xlabel=comparisons[0][1] if index==parametric_index else (r'$t$ (sampled)' if index==3 else r'$\tau$'),
                       ylabel=comparisons[1][1] if index==parametric_index else comparisons[index][1])
                if numerical_check:
                    ax.set_ylim(bottom=0)
                ax.legend()
        fig.tight_layout()
        name = comparisons[index][0] if index<parametric_index else 'parametric'
        prefix = 'MFIM_L_sweep' if model=='MFIM' else 'N_g_sweep'
        fig.savefig(os.path.join(ut.FIG_DIR,f'{prefix}_{name}.png'),**ut.FIG_SAVE_OPTIONS)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',nargs='+',required=True,help='Production results_*.npz archives')
    parser.add_argument('--gap-lines',action='store_true')
    parser.add_argument('--thermalization-fraction',type=float)
    args = parser.parse_args()
    if args.thermalization_fraction is not None and not 0<args.thermalization_fraction<1:
        parser.error('--thermalization-fraction must lie between zero and one')
    if args.results:
        plot_archives(args.results, gap_lines=args.gap_lines, fraction=args.thermalization_fraction)
        raise SystemExit(0)
    # No arguments retain the original .npy cache plotting workflow.
    # Sweep Parameters
    N_series = np.arange(18, 24+2, 2)
    g_series = np.array([0, 1, 10, 100], dtype=float)
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
                temp_rp_2norm[N_idx, g_idx] = np.nan
                proj_scr_2norm[N_idx, g_idx] = np.nan

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

