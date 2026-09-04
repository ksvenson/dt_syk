import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np
import dynamite.computations as comp
import itertools as it

import utilities as ut


@ut.cache('npy', 'time_evolved_overlaps')
def time_evolved_overlaps(H, psi0, dt, num_samples, note=None):
    """
    Computes < psi0 | psi(t) > in `dt` increments.
    The first element of the returned array is < psi0 | psi_0 > = 1.
    The last element of the returned array is < psi0 | psi(dt * num_samples) >.

    Parameters:
    `H` (dynamite operator): the hamiltonian
    `psi0` (dynamite state): initial state to evolve
    `dt` (float): time step between each evolved state.
    `num_samples` (int): number of overlaps to compute.

    Returns:
    Array of length `num_samples`.
    """
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

@ut.cache('npy', 'temp_square_2norm_exact')
def temp_square_2norm_exact(evals, pops, k, tau_series, comm=None, chunk=2**8, verbose=True, note=None):
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
    multi_sets, mult = np.unique(np.sort(list(np.ndindex((d,)*k)), axis=-1), axis=0, return_counts=True)  # (D_k, k), (D_k,)
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

@ut.cache('npz', 'eng_diffs')
def eng_diffs(evals, k, bin_edges, comm=None, chunk=1024, verbose=True, note=None):
    combs = list(it.combinations(np.arange(evals.size), k))
    eng_sum = np.sum(evals[combs], axis=-1)
    local_min = np.max(evals)

    rank = 0
    size = 1
    if comm:
        rank = comm.Get_rank()
        size = comm.Get_size()
    c = -1
    
    local_counts = np.zeros(bin_edges.size-1, dtype=np.int32)
    for i in range(0, len(combs), chunk):
        for j in range(i, len(combs), chunk):
            c += 1
            if c % size != rank:
                continue

            if verbose:
                print(f'rank {rank:03d}/{size:03d}: ({i}, {j}) / {len(combs)}')

            diffs = np.abs(np.subtract.outer(eng_sum[i:i+chunk], eng_sum[j:j+chunk]))  # (chunk, chunk)
            if i == j:
                diffs = diffs[np.triu_indices(diffs.shape[0], k=1)]
            else:
                diffs = diffs.ravel()
            local_counts += np.histogram(diffs, bins=bin_edges)[0]

            new_min = np.min(diffs)
            if new_min < local_min:
                local_min = new_min
    # print(f'sum: {np.sum(local_counts)}')
    # print(f'len: {(len(combs)**2 - len(combs)) // 2}')
    # assert np.sum(local_counts) == ((len(combs)**2 - len(combs)) // 2)
    # print(f'k={k} min diff: {local_min}')
    if comm:
        counts = np.empty_like(local_counts)
        m = 0
        comm.Allreduce(local_counts, counts)
        comm.allreduce(local_min, m)
        return {'counts': counts, 'min': m}
    else:
        return {'counts': local_counts, 'min': local_min}

