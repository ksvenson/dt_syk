import numpy as np
import scipy as sp

import utilities as ut


@ut.cache('npz', 'eigs')
def eig_system(mat, note=None):
    evals, evecs = sp.linalg.eigh(mat)
    # return in this format to make compatible with caching
    return {'evals': evals, 'evecs': evecs}

def get_pops(evecs, psi0):
    pops = np.einsum('ab,b->a', evecs.conj().T, psi0)
    return np.abs(pops)**2

