import numpy as np
import itertools as it
import math

import utilities as ut


@ut.cache('npy', 'time_evolved_overlaps')
def time_evolved_overlaps(H, psi0, dt, num_samples, note=None):
    """
    Computes < psi0 | psi(t) > in `dt` increments.
    The first element of the returned array is < psi0 | psi_0 > = 1.
    The last element of the returned array is < psi0 | psi(dt * (num_samples - 1)) >.

    Parameters:
    `H` (dynamite operator): the hamiltonian
    `psi0` (dynamite state): initial state to evolve
    `dt` (float): time step between each evolved state.
    `num_samples` (int): number of overlaps to compute.

    Returns:
    Array of length `num_samples`.
    """
    from dynamite import computations as comp

    overlaps = np.full(num_samples, np.nan, dtype='complex')
    psit = psi0.copy()
    overlaps[0] = 1
    for i in np.arange(1, num_samples):
        print(f'{i+1} / {num_samples}')
        psit = comp.evolve(H, psit, dt)
        overlaps[i] = psi0.dot(psit)
    return overlaps


def temp_square_2norm_sampled(overlaps, k_series):
    """Calculate the purity of the original discrete temporal ensemble.

    If the supplied overlaps are A(l*dt)=<psi(0)|psi(l*dt)>, the returned
    entry at index n-1 is the purity of the uniform ensemble containing
    n states at times 0,dt,...,(n-1)*dt:

        P_n,k = 1/n + (2/n**2) sum_{l=1}^{n-1} (n-l)|A(l*dt)|**(2k).

    Parameters
    ----------
    overlaps : complex one-dimensional array
        Uniformly spaced return amplitudes, with overlaps[0]=1.
        Only their magnitudes are used, so an overall phase is irrelevant.
    k_series : one-dimensional integer array
        Requested moment orders, in the desired output-column order.

    Returns
    -------
    real array, shape (len(overlaps), len(k_series))
        Tr[(rho_discrete^(k))**2], not a moment matrix or its square.
        The first row is one. Subtract the matching RP purity and take a
        square root to obtain the discrete temporal/RP distance.

    This is retained for the earlier regular-grid observable. The production
    continuous-time curve uses temporal_distance_continuous instead. It is
    a different averaging prescription at finite dt, not a different way
    of exactly evaluating the same finite-time continuous integral."""
    assert np.abs(overlaps[0] - 1) < 10**-10
    ret = np.full((overlaps.size, k_series.size), np.nan)
    ret[0] = 1
    ret[1:] = np.power.outer(np.abs(overlaps[1:]), 2*k_series)
    helper = np.arange(1, overlaps.size)[:, np.newaxis]
    t1 = np.cumsum(ret[1:], axis=0)
    t2 = np.cumsum(ret[1:] * helper, axis=0)
    ret[1:] = (1 + 2*t1) / (helper+1) - 2*t2 / (helper+1)**2
    return ret

@ut.cache('npy', 'temp_square_2norm_exact_v2')
def temp_square_2norm_exact(evals, pops, k, tau_series, comm=None, chunk=2**8, verbose=True, note=None, max_sums=2**20):
    """Evaluate continuous temporal/RP squared distances by energy-sum pairs.

    This is an independent small-system alternative to overlap integration.
    A label M contains k energy indices with repetition allowed and without
    ordering. If index alpha occurs n_alpha times, define

        S_M = sum_alpha n_alpha E_alpha,
        w_M = (k!/product_alpha n_alpha!)*product_alpha p_alpha**n_alpha.

    The squared Hilbert-Schmidt distance is

        2 * sum_{M<M'} w_M*w_M' * sinc((S_M-S_M')*tau/2)**2,

    with sinc(x)=sin(x)/x and sinc(0)=1. Distinct labels with equal energy
    sums are retained and can prevent the long-time distance from vanishing.
    The ordering M<M' only ensures each pair is counted once. It does not
    assert that their energies are unequal.

    Parameters
    ----------
    evals, pops : real one-dimensional arrays of the same length
        Energies and nonnegative normalized populations in their eigenbasis.
    k : positive integer
        Moment order.
    tau_series : real one-dimensional array
        Times at which to evaluate the exact finite sum. No integration grid
        is used and these times need not be uniformly spaced.
    comm : MPI communicator or None
        Optional processes sharing the pair sums; their results are added.
    chunk : positive integer, default 256
        Number of sum labels handled together, limiting temporary storage.
    verbose : bool
        Print progress if True.
    note : str or None
        Name identifying saved results; change it whenever inputs change.
    max_sums : integer or None, default 2**20
        Refuse to enumerate more than this many labels. None removes the guard.

    Returns
    -------
    real array, shape tau_series.shape
        Tr[(rho_T^(k)-rho_RP^(k))**2], the squared distance. Take its square
        root to compare with the 'distance' output of temporal_distance_continuous.

    The sum is exact up to floating-point arithmetic, but the number of labels
    is binomial(len(evals)+k-1,k) and the pair count is roughly its square.
    It is unsuitable for large N even with the improved label construction."""
    partial = np.zeros(tau_series.shape)
    d = evals.size
    # Let D_k be the dimension of the symmetric subspace
    n_sums = math.comb(d+k-1, k)
    if max_sums is not None and n_sums > max_sums:
        raise ValueError(f'{n_sums} multisets require quadratic pair work; use temporal_distance_continuous')
    multi_sets = np.array(list(it.combinations_with_replacement(range(d), k)))
    mult = np.array([math.factorial(k)/math.prod(
        math.factorial(c) for c in np.bincount(m, minlength=d)) for m in multi_sets])
    eng_sum = np.sum(evals[multi_sets], axis=-1)                                                          # (D_k,)
    pop_prod = mult * np.prod(pops[multi_sets], axis=-1)                                                  # (D_k,)
    rank = 0
    size = 1
    if comm:
        rank = comm.Get_rank()
        size = comm.Get_size()
    count = -1
    for a in np.arange(0, multi_sets.shape[0], chunk):
        for b in np.arange(a, multi_sets.shape[0], chunk):  # sum over upper triangle of (a, b)
            count += 1
            if count % size != rank:
                continue
            if verbose:
                print(f'rank {rank:03d}/{size:03d}: ({a}, {b}) / {multi_sets.shape[0]}')
            sinc_arg = np.subtract.outer(  # (chunk, chunk)
                eng_sum[a : a + chunk],
                eng_sum[b : b + chunk]
            )
            coeff = np.multiply.outer(     # (chunk, chunk))
                pop_prod[a : a + chunk],
                pop_prod[b : b + chunk]
            )
            if a == b:
                mask = np.triu_indices(sinc_arg.shape[0], k=1)
                sinc_arg = sinc_arg[mask]
                coeff = coeff[mask]
            else:
                sinc_arg = sinc_arg.ravel()
                coeff = coeff.ravel()
            sinc_arg = np.multiply.outer(sinc_arg, tau_series / 2)  # (chunk**2, tau_series)
            partial += 2 * np.sum(np.sinc(sinc_arg / np.pi)**2 * coeff[:, np.newaxis], axis=0)  # (tau_series,)
    if comm:
        ret = np.empty_like(partial)
        comm.Allreduce(partial, ret)
        return ret
    else:
        return partial

@ut.cache('npz', 'eng_diffs_v2')
def eng_diffs(evals, k, bin_edges, comm=None, chunk=2**10, verbose=True,
              note=None, max_sums=2**22):
    """Count differences between distinct k-subset energy sums.

    Each sum uses k different energy indices. Two subsets that differ in
    their indices are counted as distinct even if their sums are equal.
    Each unordered pair of subsets is counted once; a subset compared with
    itself is excluded. Consequently a genuine equality of distinct sums
    contributes to the zero-difference bin and makes the minimum zero.

    Parameters
    ----------
    evals : real one-dimensional array
        Energy levels; their input order need not be sorted.
    k : positive integer, at most len(evals)
        Number of different indices in each sum. Repetition within a sum is
        excluded here, unlike in the k-copy temporal-moment formula.
    bin_edges : increasing real array
        Histogram interval boundaries. Bins include their lower boundary and
        exclude their upper boundary, except that the last upper boundary is
        included, matching numpy.histogram. Differences are absolute values.
    comm : MPI communicator or None
        Optional group of processes sharing the counting work. Every process
        constructs the same sums, and their partial histograms are added.
    chunk : positive integer, default 1024
        Number of sums handled together when counting; changes storage only.
    verbose : bool
        Print counting progress if True.
    note : str or None
        Name of saved results; change it for different levels, k, or bins.
    max_sums : integer or None, default 2**22
        Refuse to enumerate more than this many subsets. None removes the
        guard; it does not remove the potentially very large memory demand.

    Returns
    -------
    dict
        'counts': integer array of length len(bin_edges)-1.
        'min': minimum absolute difference over all distinct subset pairs,
        regardless of the histogram range; infinity if no pair exists.

    Method
    ------
    Sort the M=binomial(len(evals),k) sums. The minimum is then the smallest
    difference between neighbors. For each sum and histogram boundary, locate
    how many larger-index sums have differences below that boundary, halving
    the remaining search interval each time. This avoids explicitly listing
    all M*(M-1)/2 pairs, but enumerating M sums can still be impractical."""
    edges = np.asarray(bin_edges, dtype=float)
    if (k < 1 or k > len(evals) or chunk < 1 or edges.ndim != 1
            or len(edges) < 2 or np.any(np.diff(edges) <= 0)):
        raise ValueError('Invalid k, chunk, or histogram edges')
    M = math.comb(len(evals), k)
    if max_sums is not None and M > max_sums:
        raise ValueError(f'{M} subset sums exceed max_sums={max_sums}; histogram enumeration is too large')
    sums = np.sort(np.fromiter((np.sum(evals[list(c)])
        for c in it.combinations(range(len(evals)), k)), dtype=float, count=M))
    minimum = float(np.min(np.diff(sums))) if M > 1 else np.inf
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    for start in range(rank*chunk, M, size*chunk):
        if verbose:
            print(f'rank {rank}: histogram sums {start}/{M}')
        indices = np.arange(start, min(start+chunk, M))
        # Restrict to j>i, so zero differences between different sums survive
        # but the trivial self-comparison never enters the first bin.
        def bound(edge, inclusive=False):
            # Search actual differences rather than sums[i]+edge: the latter
            # can move a floating-point value across an exact histogram edge.
            """Find the first larger-index sum beyond a difference threshold.

            For each currently selected sum i, return the first j>i whose
            difference sums[j]-sums[i] is >=edge, or >edge when inclusive=True.
            Return M if no such j exists. This implements the histogram's
            boundary convention using the actual floating-point differences."""
            lower = indices+1
            upper = np.full(len(indices), M, dtype=np.int64)
            while np.any(lower < upper):
                middle = (lower+upper)//2
                values = sums[np.minimum(middle, M-1)]-sums[indices]
                advance = values <= edge if inclusive else values < edge
                active = lower < upper
                lower = np.where(active & advance, middle+1, lower)
                upper = np.where(active & ~advance, middle, upper)
            return lower
        lower = bound(edges[0])
        for j in range(len(edges)-1):
            upper = bound(edges[j+1], inclusive=(j == len(edges)-2))
            counts[j] += np.sum(upper-lower, dtype=np.int64)
            lower = upper
    if comm is not None:
        total = np.empty_like(counts)
        comm.Allreduce(counts, total)
        counts = total
    # All ranks construct the same sums, hence the same minimum; no reduction
    # (and no erroneous positional use of MPI's op argument) is required.
    return {'counts': counts, 'min': minimum}


def spectral_sampling_diagnostics(evals, k_max, dt):
    """Check whether a regular time spacing resolves temporal-moment frequencies.

    With hbar=1, the return amplitude is A(u)=sum_alpha p_alpha exp(-i E_alpha u).
    The frequencies in |A(u)|**(2k) are differences between two sums of k
    energies, so their absolute values cannot exceed k*W, with
    W=max(E_alpha)-min(E_alpha). A conservative regular-sampling condition is
    dt < pi/(k*W): at least two samples per shortest possible oscillation.

    Parameters
    ----------
    evals : real one-dimensional array
        Energies in the retained sector. All supplied energies are used, even
        those with negligible population; this makes the bound conservative.
    k_max : positive integer
        Largest temporal moment order that will be evaluated.
    dt : positive float
        Proposed spacing between overlap evaluations, in the same time units
        as exp(-i E t).

    Returns
    -------
    dict
        'bandwidth': W; 'moment_frequency_bound': k_max*W;
        'nyquist_dt': pi/(k_max*W), or infinity when W=0;
        'dt': the supplied spacing;
        'satisfies_bound': whether dt is strictly smaller than that bound.

    Notes
    -----
    A satisfied bound does not specify the accuracy of a numerical integral.
    Check accuracy by repeating with a smaller step. A failed bound does not
    by itself prove that a particular plotted feature is an artifact. This
    criterion applies to temporal purity, not to projected moments, whose
    conditional-state normalization can introduce additional frequencies."""
    width = float(np.ptp(evals))
    nyquist_dt = np.inf if width == 0 else np.pi/(k_max*width)
    return {'bandwidth': width, 'moment_frequency_bound': k_max*width,
            'nyquist_dt': nyquist_dt, 'dt': dt,
            'satisfies_bound': bool(dt < nyquist_dt)}


def temporal_distance_continuous(evals, pops, k_series, tau_series,
                                 dt=0.1, chunk=1024):
    """Calculate continuous-time temporal/RP moment distances from overlaps.

    Define rho_T^(k)(tau) as the uniform average of
    (|psi(t)><psi(t)|)**tensor k for 0<=t<=tau. Define rho_RP^(k) by independently
    averaging every energy coefficient's phase, keeping its magnitude fixed.
    This function evaluates their Hilbert-Schmidt distance without constructing
    either large k-copy operator.

    For A(u)=sum_alpha p_alpha exp(-i E_alpha u), the temporal purity is

        P_T,k(tau) = (2/tau**2) integral_0^tau (tau-u)|A(u)|**(2k) du.

    If P_RP,k=Tr[(rho_RP^(k))**2], then

        ||rho_T^(k)(tau)-rho_RP^(k)||_HS**2 = P_T,k(tau)-P_RP,k.

    The cross term Tr[rho_T^(k) rho_RP^(k)] equals P_RP,k at every time, which
    is why subtracting one RP purity suffices. The RP purity is evaluated
    exactly by random_phase.rpe_square_2norm. The remaining one-dimensional
    integral is approximated by composite Simpson integration: over adjacent
    intervals of width h, endpoint and midpoint values have weights 1,4,1,
    multiplied by h/3. Both integral f(u) du and integral u*f(u) du are
    accumulated so every requested reporting time uses the same overlaps.

    Parameters
    ----------
    evals : real array, shape (number_of_energy_states,)
        Energies E_alpha, in units with hbar=1.
    pops : real array of the same shape
        p_alpha=|<E_alpha|psi(0)>|**2, nonnegative and summing to one.
    k_series : one-dimensional array of positive integers
        Moment orders, in the desired output-column order.
    tau_series : real one-dimensional array
        Uniformly spaced reporting times, starting at zero, with at least
        two entries. These are not necessarily the finer integration times.
    dt : positive float, default 0.1
        Upper bound on integration spacing. Each reporting interval is
        divided into an even number of equal steps so Simpson integration
        ends exactly at each requested time. The actual spacing h is <=dt.
        Decreasing dt adds only overlap evaluations, not projected states.
    chunk : positive integer, default 1024
        Approximate maximum number of new integration points evaluated
        together, rounded down to an even number (at least two). This limits
        temporary storage without changing the integration grid.

    Returns
    -------
    dict
        'distance': real array, shape (len(tau_series), len(k_series)), containing
        Hilbert-Schmidt distances. At tau=0 the temporal moment is the initial
        pure-state moment and the distance is sqrt(1-P_RP,k).
        'distance_squared': same shape, containing the computed P_T,k-P_RP,k
        before small negative values are clipped for the square root.
        'dt_effective': scalar NumPy array containing the actual spacing h.

    Accuracy and checks
    -------------------
    Spectral phases are evaluated directly; Simpson integration is the
    approximation. Compare dt with dt/2 to assess convergence. Values below
    -1e-10 for a squared distance raise an error suggesting a smaller dt;
    remaining negative values are replaced by zero only for 'distance'.
    The calculation stores groups of overlaps rather than all evolved states.
    Shifting every energy by the same constant leaves the result unchanged.

    Energy degeneracies or equal sums of energies do not invalidate the
    distance identity. They can prevent its long-time limit from vanishing:
    equality of the continuous temporal and independent random-phase
    ensembles requires an additional condition excluding relevant resonances.
    No such condition is assumed here."""
    import random_phase as rp

    times = np.asarray(tau_series, dtype=float)
    ks = np.asarray(k_series, dtype=int)
    evals = np.asarray(evals, dtype=float)
    pops = np.asarray(pops, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('Output times must start at zero and increase')
    spacing = times[1]
    if not np.allclose(np.diff(times), spacing, rtol=1e-10, atol=1e-12):
        raise ValueError('Output times must be uniformly spaced')
    if not np.isfinite(dt) or dt <= 0 or chunk < 1 or np.any(ks < 1):
        raise ValueError('dt, chunk and moment orders must be positive')
    if (pops.shape != evals.shape or np.any(pops < 0)
            or not np.isclose(pops.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError('Populations must be nonnegative and normalized')
    # Composite Simpson quadrature needs an even number of subdivisions.
    subdivisions = max(2, 2*int(np.ceil(spacing/(2*dt))))
    h = spacing/subdivisions
    baseline = rp.rpe_square_2norm(pops, int(ks.max()))[ks-1]
    squared = np.empty((len(times), len(ks)))
    squared[0] = 1-baseline
    integral_f = np.zeros(len(ks))
    integral_uf = np.zeros(len(ks))
    previous_f = np.ones(len(ks))
    energies = evals-evals[0]
    n_steps = (len(times)-1)*subdivisions
    # Work in pairs of intervals so every output time is a Simpson endpoint.
    for start in range(1, n_steps+1, 2*max(1, chunk//2)):
        indices = np.arange(start, min(start+2*max(1, chunk//2), n_steps+1))
        u = indices*h
        overlaps = np.exp(-1j*u[:, None]*energies) @ pops
        f = np.abs(overlaps[:, None])**(2*ks)
        mid_f, end_f = f[::2], f[1::2]
        mid_u, end_u = u[::2, None], u[1::2, None]
        left_f = np.concatenate((previous_f[None, :], end_f[:-1]), axis=0)
        int_f = integral_f + np.cumsum(h/3*(left_f+4*mid_f+end_f), axis=0)
        int_uf = integral_uf + np.cumsum(h/3*(
            (end_u-2*h)*left_f+4*mid_u*mid_f+end_u*end_f), axis=0)
        endpoints = indices[1::2]
        output = endpoints % subdivisions == 0
        t = end_u[output]
        squared[endpoints[output]//subdivisions] = (
            2*(int_f[output]/t-int_uf[output]/t**2)-baseline)
        integral_f = int_f[-1]
        integral_uf = int_uf[-1]
        previous_f = end_f[-1]
    if squared.min() < -1e-10:
        raise ValueError('Quadrature produced a negative squared distance; decrease --dt')
    return {'distance': np.sqrt(np.maximum(squared, 0)),
            'distance_squared': squared, 'dt_effective': np.array(h)}
