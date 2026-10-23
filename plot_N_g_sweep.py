"""Compare saved slider runs using their actual reporting axes and metadata."""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import thermalization as th


def plot_archives(paths, gap_lines=False, fraction=None):
    """Plot all four ensemble comparisons and the original parametric comparison.

    paths is a nonempty list of production NPZ files, one per (N,g). Runs must
    agree in subsystem, moment orders, random seed, parity, projected-time
    sampling controls, Scrooge sample count, and Hamiltonian convention. Their
    realization counts may differ. Each curve averages distances over disorder;
    bands are disorder SEM, not separate Haar/time-sampling error estimates.

    gap_lines optionally shows saved finite spectral reference medians only on
    the temporal plot. fraction optionally overrides the saved observed threshold.
    The instantaneous generalized Scrooge comparison is an analytic identity for
    a pure full input and uses a linear vertical axis so exact zeros are visible.
    The single-Scrooge first moment is also exactly zero and is labeled explicitly
    if the other moments require a logarithmic vertical axis.

    Five figures are written under utilities.FIG_DIR. No states are recomputed.
    Incompatible archives or duplicate (N,g) configurations raise ValueError.
    """
    if not paths:
        raise ValueError('Supply at least one result archive')
    comparisons = [
        ('temp_rp_2norm',r'$||\rho_T^{(k)}-\rho_{RP}^{(k)}||_2$'),
        ('proj_scr_2norm',r'$\mathbb{E}_t||M_k(t)-G_k[\sigma]||_2$'),
        ('proj_single_scr_2norm',r'$\mathbb{E}_t||M_k(t)-S_k[\rho_A(t)]||_2$'),
        ('proj_instantaneous_gse_2norm',r'$\mathbb{E}_t||M_k(t)-G_k[\rho(t)]||_2$')]
    runs, seen, signature = [], set(), None
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            params = json.loads(str(data['metadata']))
            pair = (params['N'],params['g'])
            if pair in seen:
                raise ValueError(f'Multiple archives for {pair}; choose one')
            seen.add(pair)
            current = tuple(params.get(k) for k in ('NA','k_max','n_scr','seed','parity',
                'projected_sampling','projected_dt','time_seed','tmax','hamiltonian_convention'))
            if signature is not None and signature != current:
                raise ValueError('Archives mix incompatible physical or sampling parameters')
            signature = current
            runs.append((params,data['tau_series'],data['k_series'],
                         {key:data[key] for key,_ in comparisons},
                         data['moment_gap_times']))
    gs = sorted({p['g'] for p,*_ in runs})
    ks = runs[0][2]
    figures = [plt.subplots(len(ks),len(gs),squeeze=False,
                           figsize=ut.fig_size(len(gs),len(ks))) for _ in range(5)]
    for params,times,orders,values,gaps in sorted(runs,key=lambda x:x[0]['N']):
        col = gs.index(params['g'])
        means = {key:array.mean(axis=0) for key,array in values.items()}
        for row,k in enumerate(orders):
            for index,(key,label) in enumerate(comparisons):
                ax = figures[index][1][row,col]
                samples = values[key]
                mean = means[key][:,row]
                sem = samples.std(axis=0,ddof=1)[:,row]/np.sqrt(len(samples)) if len(samples)>1 else np.zeros_like(mean)
                line, = ax.plot(times[1:],mean[1:],label=f'N={params["N"]}, R={len(samples)}')
                ax.fill_between(times[1:],np.maximum(mean[1:]-sem[1:],0),mean[1:]+sem[1:],alpha=.2)
                if not np.any(mean):
                    line.set_label(f'N={params["N"]} (exactly zero)')
                th.annotate_timescales(ax,times,mean[:,None],[k],
                    gap_times=gaps[:,row:row+1] if index==0 and (gap_lines or params.get('gap_lines',False)) else None,
                    fraction=fraction if fraction is not None else params.get('thermalization_fraction'),
                    label_prefix=f'N={params["N"]}: ',colors=[line.get_color()])
            figures[4][1][row,col].plot(means['temp_rp_2norm'][1:,row],means['proj_scr_2norm'][1:,row],label=f'N={params["N"]}')
    os.makedirs(ut.FIG_DIR,exist_ok=True)
    for index,(fig,axes) in enumerate(figures):
        for row,k in enumerate(ks):
            for col,g in enumerate(gs):
                ax = axes[row,col]
                zero = index==3 or (index==2 and k==1)
                ax.set(xscale='log',yscale='linear' if zero else 'log',title=f'k={k}, g={g}',
                       xlabel=comparisons[0][1] if index==4 else r'$\tau$',
                       ylabel=comparisons[1][1] if index==4 else comparisons[index][1])
                if zero:
                    ax.set_ylim(-.01,.01)
                    ax.text(.5,.7,'Exact zero',transform=ax.transAxes,ha='center')
                ax.legend()
        fig.tight_layout()
        name = comparisons[index][0] if index<4 else 'parametric'
        fig.savefig(os.path.join(ut.FIG_DIR,f'N_g_sweep_{name}.png'),**ut.FIG_SAVE_OPTIONS)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',nargs='+',required=True,help='Production results_*.npz archives')
    parser.add_argument('--gap-lines',action='store_true')
    parser.add_argument('--thermalization-fraction',type=float)
    args = parser.parse_args()
    if args.thermalization_fraction is not None and not 0<args.thermalization_fraction<1:
        parser.error('--thermalization-fraction must lie between zero and one')
    plot_archives(args.results,args.gap_lines,args.thermalization_fraction)
