"""Spectral reference times and observed distance-threshold times (hbar=1)."""
import itertools as it
import math
import numpy as np


def minimum_moment_gap(evals, k, max_sums=5_000_000):
    """Find the smallest nontrivial difference between sums of k energies.

    A sum label is an unordered k-tuple allowing repeated indices, as required
    by a k-copy moment. Permutations of the same tuple are excluded from the
    comparison; distinct tuples with equal sums give a zero minimum. All supplied
    energies are included, without a population threshold. The caller may omit
    exactly unoccupied levels before calling this function.

    Sort the M=binomial(len(evals)+k-1,k) sums. The minimum difference is between
    neighboring sorted sums, avoiding M squared pair comparisons. k=2 constructs
    sums one row at a time. Higher orders enumerate the tuples explicitly.
    max_sums guards their memory requirement, and skipped orders are reported.

    Returns a dictionary with status, n_sums, min_gap, reference_time, and
    resolution. reference_time=2*pi/min_gap is the first zero of the slowest
    individual sinc-squared term in continuous temporal purity. It is a spectral
    reference, not a prediction of when the full ensemble has thermalized.
    Zero or numerically unresolved gaps have no finite reference_time.
    resolution is a conservative floating-point scale for differences of sums;
    this is not a rigorous error bound on the diagonalization. Resolve such
    gaps with higher precision before interpreting their inverse.
    """
    energies = np.asarray(evals, dtype=float)
    if (energies.ndim != 1 or not np.all(np.isfinite(energies)) or k < 1
            or max_sums < 1):
        raise ValueError('Need finite energies, positive k and max_sums')
    M = math.comb(len(energies)+k-1, k) if len(energies) else 0
    result = {'status': 'skipped', 'n_sums': M, 'min_gap': None,
              'reference_time': None, 'resolution': None}
    if M > max_sums:
        return result
    if M < 2:
        result['status'] = 'no_pair'
        return result
    # A common energy shift cannot change gaps and reduces cancellation.
    energies = energies-energies[0]
    if k == 1:
        sums = energies.copy()
    elif k == 2:
        sums = np.empty(M)
        start = 0
        for i in range(len(energies)):
            row = energies[i]+energies[i:]
            sums[start:start+len(row)] = row
            start += len(row)
    else:
        sums = np.fromiter((sum(energies[i] for i in indices)
            for indices in it.combinations_with_replacement(range(len(energies)), k)),
            dtype=float, count=M)
    sums.sort()
    gap = float(np.min(np.diff(sums)))
    resolution = float(8*np.finfo(float).eps*max(1., np.max(np.abs(sums))))
    result.update(min_gap=gap, resolution=resolution)
    if gap <= resolution:
        result['status'] = 'zero_or_unresolved'
    else:
        result.update(status='computed', reference_time=float(2*np.pi/gap))
    return result


def threshold_time(times, distances, fraction):
    """Find an observed sustained crossing of a relative distance threshold.

    Return the first reporting time t>0 at which distance <=fraction*distance(0)
    and stays below it at every remaining sampled reporting time. This describes
    the finite observation window only; it does not guarantee future convergence.
    The distance may be temporal/RP or the running projected/Scrooge distance,
    but those are different observables and should be compared separately.

    times and distances must be finite, one-dimensional arrays of equal length,
    with strictly increasing times from zero and nonnegative distances.
    fraction must lie strictly between zero and one. Return None if the threshold
    is not reached and maintained in the window, or distance(0)=0. No assumption
    about a decay law, noise floor, or asymptotic plateau is made. The time is a
    reporting-grid endpoint; no unobserved crossing is inferred between points.
    """
    times, distances = np.asarray(times), np.asarray(distances)
    if (times.ndim != 1 or distances.shape != times.shape or len(times) < 2
            or times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(distances))
            or np.any(distances < 0) or not 0 < fraction < 1):
        raise ValueError('Invalid times, distances or fraction')
    if distances[0] == 0:
        return None
    suffix_max = np.maximum.accumulate(distances[::-1])[::-1]
    hits = np.flatnonzero(suffix_max[1:] <= fraction*distances[0])+1
    return float(times[hits[0]]) if len(hits) else None


def annotate_timescales(ax, times, mean_distances, ks, gap_times=None,
                        fraction=None, label_prefix='', colors=None):
    """Draw requested reference lines inside the observed time window.

    mean_distances has shape (reporting_times, moment_orders). gap_times, if
    supplied, has one row per realization and one column per moment order;
    unresolved or skipped references are NaN. A dotted line shows the median
    over finite, resolved gap times, rather than an inverse gap from an averaged
    spectrum. No line is shown for a column with no resolved values. A dashed
    line marks the observed threshold_time of the mean curve if fraction is set.
    Spectral lines are intended for temporal plots, not projected normalization.
    Lines outside the displayed positive-time window are omitted. Existing curve
    colors may be passed for sweep plots. Returns threshold times (NaN if absent).
    """
    crossing = np.full(len(ks), np.nan)
    for j, k in enumerate(ks):
        color = f'C{j}' if colors is None else colors[j]
        if gap_times is not None:
            finite = np.asarray(gap_times)[:, j]
            finite = finite[np.isfinite(finite) & (finite > 0)]
            if len(finite):
                ref = float(np.median(finite))
                if times[1] <= ref <= times[-1]:
                    ax.axvline(ref, color=color, linestyle=':',
                               label=f'{label_prefix}median gap reference, k={k}')
        if fraction is not None:
            t = threshold_time(times, mean_distances[:, j], fraction)
            if t is not None:
                crossing[j] = t
                ax.axvline(t, color=color, linestyle='--',
                           label=f'{label_prefix}t({fraction:g}), k={k}')
    return crossing
