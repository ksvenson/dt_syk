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
    """
    Computes the square of the kth moment of the finite-time temporal ensemble,
    $(\rho_{Temp.}^{(k)}(\tau))^2$, by directly sampling the ensemble.

    Parameters:
    `overlaps` (1d array): output of `time_evolved_overlaps`. The first element must be 1.
    `k_series` (1d array): moments of the ensemble to compute.

    Returns:
    Array of shape (`overlaps.size`, `k_series.size`).
    The first axis is the tau axis.
    Tau can be constructed with `dt * np.arange(overlaps.size)`.
    """
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
    """
    Computes the difference between the kth moments of the squares of finite-time
    temporal ensemble and random phase ensemble:
    $(\rho_{Temp.}^{(k)}(\tau))^2 - (\rho_{RP.}^{(k)})^2$.
    Instead of sampling the ensembles, this method uses the closed-form expression (eq
    34) in Mark et al.: https://arxiv.org/pdf/2403.11970

    Parameters:
    `evals` (1d array): all eigenvalues of the Hamiltonian.
    `pops` (1d array): eigenspace populations of the initial state: |< psi0 | E_n >|^2
    `k` (int): moment of the ensemble to compute
    `tau_series` (1d array): times at which to comptue the ensemble.
    `comm` (mpi4py communicator object): .
    `chunk` (int): size of blocks summed over. Make smaller to reduce memory usage.
    """
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
    """Histogram of unordered pairs of distinct, no-repetition k-subset sums.

    Sort sums, then count differences with vectorized binary searches, one per
    histogram edge. This keeps numpy.histogram's last-edge inclusion convention.
    Repeated indices within each sum remain excluded, as in the original API.
    The minimum comes from adjacent sorted sums; distinct equal sums return 0.
    Cost: O(M log M + bins*M log M), memory O(M), M=binomial(D,k).
    The combination count is still prohibitive for large D and k.
    """
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
    """Conservative frequency bound for |sum p exp(-i E t)|^(2k)."""
    width = float(np.ptp(evals))
    nyquist_dt = np.inf if width == 0 else np.pi/(k_max*width)
    return {'bandwidth': width, 'moment_frequency_bound': k_max*width,
            'nyquist_dt': nyquist_dt, 'dt': dt,
            'satisfies_bound': bool(dt < nyquist_dt)}


def temporal_distance_continuous(evals, pops, k_series, tau_series,
                                 dt=0.1, chunk=1024):
    """Continuous temporal/RP Frobenius distances by streamed Simpson quadrature.

    Purity = 2/tau^2 integral_0^tau (tau-u)|<psi(0)|psi(u)>|^(2k) du.
    Only spectral overlaps are evaluated, at O(D) per node, independently of
    the projected-state sampling. Output times must be a uniform grid from 0.
    dt is an upper bound: subdivisions are rounded up to hit output times.
    This is quadrature, not an exact analytic temporal calculation. Refine dt
    to verify convergence. RP populations need not obey a nonresonance assumption
    for the distance identity; resonances can leave a nonzero limiting distance.
    """
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
