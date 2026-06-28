import dynamite as dm
import dynamite.states as st
import numpy as np
import scipy as sp
import itertools as it

import hamiltonian_factory as hf
import linalg as la

import utilities as ut


def rpe_moment(pops, k):
    d = 2**dm.config.L
    ret = np.zeros((d,)*2*k)
    for idx in np.ndindex((d,)*k):
        perms = set(it.permutations(idx))  # only include unique permtations
        for perm in perms:
            ret[idx + perm] = np.prod(pops[list(idx)])
    return ret.reshape((d**k, d**k))

def rpe_square_2norm(pops, k):
    d = 2**dm.config.L
    ret = 0
    for idx in np.ndindex((d,)*k):
        ret += len(set(it.permutations(idx))) * np.prod(pops[list(idx)])**2
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

    
