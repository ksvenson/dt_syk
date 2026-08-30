import dynamite as dm
import dynamite.states as st
import numpy as np
import scipy as sp
import itertools as it

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


@ut.cache('npy', 'rpe_square_2norm')
def rpe_square_2norm(pops, k, chunk=2**12, note=None):
    ret = np.zeros(1)  # numpy array so we can cache it
    multi_sets = np.array(list(it.combinations_with_replacement(np.arange(pops.size), k)))
    mult = np.full(multi_sets.shape[0], sp.special.factorial(k))
    for m_idx, m in enumerate(multi_sets):
        _, counts = np.unique(m, axis=-1, return_counts=True)
        mult[m_idx] /= np.prod(sp.special.factorial(counts))
    for c_idx in range(0, multi_sets.shape[0], chunk):
        ret += np.sum((mult[c_idx:c_idx+chunk] * np.prod(pops[multi_sets[c_idx:c_idx+chunk]], axis=-1))**2)
    return ret


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

    
