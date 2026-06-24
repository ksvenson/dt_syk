import dynamite as dt
import dynamite.states as st
import numpy as np
import scipy as sp
import itertools as it

import hamiltonian_factory as hf

import utilities as ut


def eig_system(mat):
    evals, evecs = sp.linalg.eigh(mat)
    # return in this format to make compatible with caching
    return {'evals': evals, 'evecs': evecs}

def get_pops(evecs, psi0):
    overlaps = np.einsum('ab,b->a', evecs.conj().T, psi0)
    return np.abs(overlaps)**2

def rpe_moment(pops, k):
    d = 2**d.config.L
    ret = np.zeros((d,)*2*k)
    for idx in np.ndindex((d,)*k):
        perms = set(it.permutations(idx))  # only include unique permtations
        for perm in perms:
            ret[idx + perm] = np.prod(pops[list(idx)])
    return ret.reshape((d**k, d**k))

def rpe_square_2norm(pops, k):
    d = 2**dt.config.L
    ret = 0
    for idx in np.ndindex((d,)*k):
        ret += len(set(it.permutations(idx))) * np.prod(pops[list(idx)])**2
    return ret


if __name__ == '__main__':
    dt.config.L = 6
    k = 2
    
    H = hf.MFIM().to_numpy(sparse=False)
    psi0 = st.State(0).to_numpy()

    eigs = eig_system(H)
    pops = get_pops(eigs['evecs'], psi0)
    np.save('./pops.npy', pops)
    
