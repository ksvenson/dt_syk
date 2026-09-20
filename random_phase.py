import dynamite as dm
import dynamite.states as st
import numpy as np
import scipy as sp
import itertools as it
import math

import hamiltonian_factory as hf
import linalg as la

import utilities as ut


def rpe_moment(pops, k):
    d = pops.size
    ret = np.zeros((d,)*2*k)
    for idx in np.ndindex((d,)*k):
        perms = set(it.permutations(idx))  # only include unique permtations
        for perm in perms:
            ret[idx + perm] = np.prod(pops[list(idx)])
    return ret.reshape((d**k, d**k))


# @ut.cache('npy', 'rpe_square_2norm')
def rpe_square_2norm_old(pops, k, chunk=2**12, note=None):
    ret = 0
    # multi_sets = np.array(list(it.combinations_with_replacement(np.arange(pops.size), k)))
    # mult = np.full(multi_sets.shape[0], sp.special.factorial(k))
    # for m_idx, m in enumerate(multi_sets):
    #     _, counts = np.unique(m, axis=-1, return_counts=True)
    #     mult[m_idx] /= np.prod(sp.special.factorial(counts))
    # for c_idx in range(0, multi_sets.shape[0], chunk):
    #     ret += np.sum((mult[c_idx:c_idx+chunk] * np.prod(pops[multi_sets[c_idx:c_idx+chunk]], axis=-1))**2)
    for m in np.ndindex((pops.size,)*k):
        ret += len(set(it.permutations(m))) * np.prod(pops[list(m)])**2
    return np.array(ret)  # numpy array so we can cache it


def log_coeffs(k):
    """
    Computes {a_0, ..., a_k}, where log(sum_m z^m/(m!)^2) = sum_{j} a_j z^j.
    This is just a formal series expansion, independent of any data.
    The recurrence relation used for this can be found as follows. Define:
    G(z) = sum_{m} z^m/(m!)^2
    A(z) = sum_{j} a_j z^j
    We want G(z) = exp(A(z)). Then we can write:
    G'(z) = A'(z)*G(z)
    By comparing powers of z, we can find the coefficients {a_0, ..., a_k}.
    Credit: Claude Opus 5
    """
    g = [1/math.factorial(m)**2 for m in range(k+1)]
    a = np.zeros(k+1)
    for n in range(1, k+1):
        a[n] = g[n] - sum(j*a[j]*g[n-j] for j in range(1, n))/n
    return a


def rpe_square_2norm(pops, k):
    """
    Computes the trace of the square of the 1st through kth moments of the random phase
    ensemble.
    Call this quantity T_1, ..., T_k.
    Returns T_1, ..., T_k as a length k array.

    We can write:
    T_k = sum_{{M}} mult(M)^2 prod_n q_n^{c_n(M)}
    where
     - {M} is the set of length-k multisets drawn from {1, ..., D}. A length-k multiset
       is an unordered collection of k elements with repetition allowed. D = len(pops)
       is the dimension of the Hilbert space.
     - mult(M) is the multinomial coefficient associated with M.
     - q_n = pops[n]^2
     - c_n(M) is the number of occurrences of n in M.

    T_k is computed as follows. First, notice that {M} is in bijection with tuples of
    non-negative integers (c_1, ..., c_D) such that c_1 + ... + c_D = k. Define the
    generating function:
    G(z) = sum_{c >= 0} z^c / (c!)^2.
    The factor (c!)^2 comes from expanding mult(M)^2. T_k is then (k!)^2 times the
    coefficient of z^k in prod_{n=1}^D G(q_n z). This is written:
    T_k = (k!)^2 [z^k] prod_{n=1}^D G(q_n z).
    
    To find this coefficient, we can first Taylor expand log(G(z)):
    log(G(z)) = sum_j a_j z^j.
    Then:
    log(prod_{n=1}^D G(q_n z)) = sum_{n=1}^D sum_j a_j (q_n z)^j
                               = sum_j a_j P_j z^j
    where P_j := sum_{n=1}^D q_n^j. Now:
    T_k = (k!)^2 [z^k] prod_{n=1}^D G(q_n z).
        = (k!)^2 [z^k] exp(log(prod_{n=1}^D G(q_n z)))
        = (k!)^2 [z^k] exp(sum_j a_j P_j z^j)
    The coefficients a_j are independent of `pops` and are computed in `log_coeffs`.
    The coefficients of exp(sum_j a_j P_j z^j) are computed similarly in this function.

    Note: after computing T_k, we get T_1, ..., T_{k-1} for free just by looking at the
    lower order coefficients.
    
    Credit: Claude Opus 5
    """
    # Compute a_j's
    a = log_coeffs(k)

    # Compute P_j's
    P = np.ones(k+1)
    for j in range(1, k+1):
        P[j] = np.sum(pops**(2*j))
    
    # c_j = a_j P_j
    c = a * P

    # Compute coefficients of T_k
    e = np.zeros(k+1)
    e[0] = 1.0
    for n in range(1, k+1):
        e[n] = sum(j*c[j]*e[n-j] for j in range(1, n+1))/n
    return np.array([math.factorial(n)**2 for n in range(1, k+1)]) * e[1:]


if __name__ == '__main__':
    dm.config.L = 6
    k = 2
    
    H = hf.MFIM().to_numpy(sparse=False)
    psi0 = st.State(0).to_numpy()

    eigs = la.eig_system(H)
    pops = la.get_pops(eigs['evecs'], psi0)

    occ = np.sort(eigs['evals'][pops > 10e-9])
    evals = np.sort(eigs['evals'])
    print(f'Dimension: {2**dm.config.L}')
    print(f'num occ: {occ.size}')
    print(f'min gap in occ: {np.min(np.diff(occ))}')
    print(f'min gap overall: {np.min(np.diff(evals))}')
    print(f'max gap in occ: {occ[-1] - occ[0]}')
    print(f'max gap overall: {evals[-1] - evals[0]}')
    print(f'shortest period: {2 * np.pi / (evals[-1] - evals[0])}')

    moment = rpe_moment(pops, k)

    print('Trace of operator squared:')
    print(np.einsum('ab,ba->', moment, moment))
    print(np.trace(moment @ moment))
    print('Our answer:')
    print(rpe_square_2norm(pops, k))

    
