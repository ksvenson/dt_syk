"""Shared ensemble-analysis controls, plotting, and time-averaging conventions.

Both Hamiltonian drivers use these routines. All state evolution and moment
calculations remain in temporal_funcs and projected_funcs.
"""
import argparse
import os
import numpy as np
import utilities as ut
import thermalization as th
import temporal_funcs as tf

def positive_float(arg):
    """Read a finite, strictly positive real-valued analysis parameter.

    Used for time spacings and the maximum evolution time. Zero, negative
    values, NaN, and infinity are rejected before the calculation starts."""
    value = float(arg)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return value


def positive_int(arg):
    """Read a strictly positive integer analysis parameter.

    Used for sample counts, moment orders, and numbers of times handled
    together. Invalid values are rejected before the calculation starts."""
    value = int(arg)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def plot_results(times, temporal, projected, ks, path, title, gap_times=None, fraction=None,
                 single=None, single_instant=None, sample_times=None):
    """Plot disorder-averaged temporal and projected distances.

    Parameters
    ----------
    times : real array, shape (number_of_reporting_times,)
        The saved endpoint times; time zero is omitted from logarithmic plots.
    temporal, projected : real arrays, shape (realizations, times, moment_orders)
        Temporal/RP distance and running mean of per-state projected/Scrooge
        distance, respectively, as calculated by main.
    ks : array of integers
        Moment order corresponding to each final array index.
    path : str
        Filename label appended to each figure name; not a directory path.
    title : str
        Figure title describing the run.
    gap_times : array, shape (realizations, moment_orders), or None
        Resolved spectral reference times 2*pi/minimum k-sum gap. The temporal
        plot can show their finite median; skipped/unresolved entries are NaN.
        These reference lines are not predictions of the projected timescale.
    fraction : float in (0,1), or None
        If supplied, mark observed sustained crossings below this fraction
        of the initial mean distance, separately for each curve.

    single : array or None
        Optional running distances to single Scrooge(rho_A(t)), with the same
        shape as projected. Its k=1 column is a computed numerical residual,
        mathematically zero, and is plotted without replacing its values.
    single_instant : array or None
        Single-Scrooge distance at each sampled time, before time averaging.
        Disorder averages may still be shown. sample_times gives the actual
        sampled times with shape (realizations,times); otherwise use times.

    Up to four PNG figures are saved under utilities.FIG_DIR. Shading is the standard
    error across disorder realizations, with zero shading for a single
    realization. It is not a separate estimate of Scrooge or time-sampling
    uncertainty. This function does not recompute states or moments."""
    import matplotlib.pyplot as plt
    os.makedirs(ut.FIG_DIR, exist_ok=True)
    comparisons = [
        (temporal, r'$||\rho_{Temp}^{(k)}-\rho_{RP}^{(k)}||_2$', 'temp_rp_2norm'),
        (projected, r'$\mathbb{E}_{t}\,||M_k(t)-G_k[\sigma]||_2$', 'proj_scr_2norm'),
    ]
    if single is not None:
        comparisons.append((single,r'$\mathbb{E}_{t}\,||M_k(t)-S_k[\rho_A(t)]||_2$','proj_single_scr_2norm'))
    if single_instant is not None:
        comparisons.append((single_instant,r'$||M_k(t)-S_k[\rho_A(t)]||_2$','proj_single_scr_instant_2norm'))
    for data, label, prefix in comparisons:
        unaveraged = prefix == 'proj_single_scr_instant_2norm'
        x = np.mean(sample_times,axis=0) if unaveraged and sample_times is not None else times
        mean = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1)/np.sqrt(len(data)) if len(data) > 1 else np.zeros_like(mean)
        fig, ax = plt.subplots()
        for j, k in enumerate(ks):
            ax.plot(x[1:], mean[1:, j], label=f'k={k}')
            ax.fill_between(x[1:], np.maximum(mean[1:, j]-sem[1:, j], 0), mean[1:, j]+sem[1:, j], alpha=.3)
        th.annotate_timescales(ax, x, mean, ks,
                              gap_times=gap_times if prefix == 'temp_rp_2norm' else None,
                              fraction=fraction)
        # Linear scaling exposes zero and roundoff residuals without imposing
        # a fixed vertical range that could conceal a numerical discrepancy.
        numerical_check = prefix in ('proj_single_scr_2norm','proj_single_scr_instant_2norm')
        ax.set(xlabel=r'$t$ (sampled)' if unaveraged else r'$\tau$', ylabel=label, xscale='log',
               yscale='linear' if numerical_check or not np.any(mean[1:] > 0) else 'log')
        if ax.get_yscale() == 'linear':
            ax.set_ylim(bottom=0)
        ax.legend()
        fig.suptitle(title)
        fig.savefig(os.path.join(ut.FIG_DIR, prefix+'_'+path+'.png'), **ut.FIG_SAVE_OPTIONS)
        plt.close(fig)


def add_sampling_arguments(parser):
    """Add common temporal, projected, Scrooge, cache and diagnostic controls."""
    parser.add_argument('--dt', default='auto', help='Temporal quadrature max step: auto or a positive number')
    parser.add_argument('--projected-dt', type=positive_float, default=1., help='Projected stratum width; independent of --dt')
    parser.add_argument('--projected-sampling', choices=('stratified', 'grid'), default='stratified', help='stratified: one uniform random time per interval; grid: regular endpoints including zero')
    parser.add_argument('--tmax', type=positive_float, default=1e5, help='Maximum evolution/reporting time, hbar=1')
    parser.add_argument('--n-scr', type=positive_int, default=1000, help='Haar inputs for higher Scrooge moments; k=1 is exact')
    parser.add_argument('--k-max', type=positive_int, default=3, help='Calculate moment orders 1 through this value')
    parser.add_argument('--chunk', type=positive_int, default=32, help='Projected time batch size')
    parser.add_argument('--seed', type=int, default=628**3, help='Base seed for disorder, Haar states and time sampling')
    parser.add_argument('--time-seed', type=int, default=0, help='Independent stratified sampling replicate')
    parser.add_argument('--check-dt', action='store_true', help='Also compute dt/2 and record convergence error')
    parser.add_argument('--cache-dir', default=ut.CACHE_DIR, help='Explicit intermediate/results directory; leaves utilities.CACHE_DIR unchanged')
    parser.add_argument('--no-plots', action='store_true', help='Save results and diagnostics without rendering figures')
    parser.add_argument('--verbose', action='store_true', help='Print per-realization diagnostics from all ranks; detailed data is always saved')
    parser.add_argument('--cache-hamiltonian', action=argparse.BooleanOptionalAction, default=False,
                        help='Read/write dense H files; unnecessary when an eigensystem is saved')
    parser.add_argument('--cache-eigensystem', action=argparse.BooleanOptionalAction, default=True,
                        help='Read/write eigenvalues and eigenvectors for repeat analysis')
    parser.add_argument('--cache-projected', action=argparse.BooleanOptionalAction, default=False,
                        help='Save per-state projected intermediate arrays in addition to final results')
    parser.add_argument('--degeneracy-tolerance', type=float, default=None,
                        help='Absolute tolerance for adapting degenerate eigenvectors to psi0; grouped energies use a common mean; default uses floating-point precision; 0 requires exact equality')
    parser.add_argument('--gap-k-max', type=int, default=1,
                        help='Exact spectral gap diagnostics through this order; 0 disables; capped by gap-max-sums')
    parser.add_argument('--gap-max-sums', type=positive_int, default=5_000_000,
                        help='Maximum number of k-sums to enumerate for each gap diagnostic')
    parser.add_argument('--gap-lines', action='store_true',
                        help='Show in-window median 2*pi/minimum-gap references on temporal plots')
    parser.add_argument('--thermalization-fraction', type=positive_float, default=None,
                        help='Mark sustained finite-window crossings below this fraction of the initial mean distance (0<f<1)')


def validate_sampling_arguments(parser, args):
    """Validate controls shared by both Hamiltonians before MPI work begins."""
    if args.seed < 0 or args.time_seed < 0:
        parser.error('seeds must be nonnegative')
    if args.dt != 'auto':
        try:
            args.dt = positive_float(args.dt)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error('--dt must be auto or a finite positive number')
    if args.degeneracy_tolerance is not None and (
            not np.isfinite(args.degeneracy_tolerance) or args.degeneracy_tolerance < 0):
        parser.error('--degeneracy-tolerance must be finite and nonnegative')
    if args.gap_k_max < 0:
        parser.error('--gap-k-max must be nonnegative')
    if args.thermalization_fraction is not None and args.thermalization_fraction >= 1:
        parser.error('--thermalization-fraction must be smaller than one')


def projected_times(times, sampling, seed, time_seed=0, realization=0):
    """Choose one state time per reporting interval, with time zero separate.

    For stratified sampling, point j>0 lies uniformly in [times[j-1],times[j]).
    Grid sampling uses the reporting endpoints. The random stream is independent
    of the Hamiltonian and Scrooge samples and matches the SYK convention.
    """
    result = np.array(times, dtype=float, copy=True)
    if sampling == 'stratified':
        rng = np.random.default_rng(np.random.SeedSequence(
            seed, spawn_key=(int(realization), 2, time_seed)))
        # Reporting grids are uniform. Retain the original SYK arithmetic and
        # random stream so existing sampled-state caches refer to the same times.
        result[1:] = times[:-1]+(times[1]-times[0])*rng.random(len(times)-1)
    elif sampling != 'grid':
        raise ValueError('sampling must be stratified or grid')
    return result


def running_distance(distances, sampling):
    """Average per-state distances with the selected temporal sampling convention.

    The input has axes (sample times, moment orders). Stratified averages exclude
    the separate t=0 point. Grid averages include that point. The raw input is
    left unchanged, so it remains available for the unaveraged fourth curve.
    """
    values = np.asarray(distances)
    if sampling == 'stratified':
        return np.concatenate((values[:1],np.cumsum(values[1:],axis=0)/
                               np.arange(1,len(values))[:,None]),axis=0)
    if sampling == 'grid':
        return np.cumsum(values,axis=0)/np.arange(1,len(values)+1)[:,None]
    raise ValueError('sampling must be stratified or grid')


def prepare_eigensystem(eigs, psi0, ks, args):
    """Adapt an eigensystem once and select the temporal integration spacing.

    Returns the phase-preserving adapted energies, eigenvectors and coefficients,
    plus the dt used and a dictionary of basis/sampling diagnostics. This defines
    one occupied initial direction per degenerate energy block. It never dephases
    psi0. The numerical block-mean energy approximation and its error bound are
    saved separately from the state-reconstruction error.
    """
    adapted = tf.initial_state_eigenbasis(eigs['evals'],eigs['evecs'],psi0,
                                         args.degeneracy_tolerance)
    evals = adapted['evals']
    pops = abs(adapted['coefficients'])**2
    max_dt = tf.spectral_sampling_diagnostics(evals,args.k_max,1.)['nyquist_dt']
    dt = min(.1,.5*max_dt) if args.dt == 'auto' else args.dt
    diagnostics = tf.spectral_sampling_diagnostics(evals,args.k_max,dt)
    diagnostics['warnings'] = []
    diagnostics['initial_state_eigenbasis'] = adapted['basis_diagnostics']
    diagnostics['initial_state_eigenbasis']['ket_error_bound_at_tmax'] = (
        args.tmax*adapted['basis_diagnostics']['max_energy_shift'])
    diagnostics['degeneracy'] = tf.degeneracy_diagnostics(evals,pops,ks,0.)
    diagnostics['degeneracy']['reference'] = adapted['basis_diagnostics']['reference']
    if dt >= max_dt:
        diagnostics['warnings'].append(
            'Temporal grid exceeds the conservative moment-frequency bound; '
            'use --dt auto or verify with --check-dt')
    adapted.update(dt=dt,diagnostics=diagnostics,pops=pops)
    return adapted


def temporal_checks(evals, pops, ks, distances, diagnostics):
    """Record the Frobenius hierarchy and initial energy-fluctuation timescales."""
    factors = np.array(diagnostics['degeneracy']['hs_partial_trace_factors'])
    diagnostics['hs_partial_trace_max_violation'] = np.maximum(
        distances[:,:1]*factors-distances,0).max(axis=0).tolist()
    centered = evals-evals[0]
    mean = float(pops@centered)
    variance = float(pops@(centered-mean)**2)
    diagnostics['energy_standard_deviation'] = float(np.sqrt(variance))
    diagnostics['short_time_overlap_widths'] = [
        float(1/np.sqrt(k*variance)) if variance > 0 else None for k in ks]


def observed_times(times, data, ks, fraction):
    """Find requested sustained threshold crossings of realization-mean curves."""
    if fraction is None:
        return np.full(len(ks),np.nan)
    mean = data.mean(axis=0)
    return np.array([np.nan if (t := th.threshold_time(times,mean[:,j],fraction)) is None
                     else t for j in range(len(ks))])
