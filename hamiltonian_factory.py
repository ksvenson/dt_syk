import numpy as np
import math
import dynamite as dm
import dynamite.operators as op
import dynamite.extras
import itertools as it

import utilities as ut


def MFIM(hx=0.9045, hz=0.8090, J=1, boundary='open'):
    """
    Mixed field Ising model.
    L should be set with `dynamite.config.L`

    Default values come from  10.1103/PhysRevE.90.052105.
    """
    # separate field and interaction terms so that boundary conditions can be applied
    # properly.
    field = op.index_sum(hx * op.sigmax() + hz * op.sigmaz())
    interaction = op.index_sum(J * op.sigmaz(i=0) * op.sigmaz(i=1), boundary=boundary)
    return field + interaction


def SYK(q, rng, J=1):
    """
    Full SYK.
    `dynamite.config.L` must already be set.

    Couplings should have length N choose q.
    """
    N = 2 * dm.config.L
    hyperedges = list(it.combinations(np.arange(N), q))
    # Couplings from https://arxiv.org/pdf/1804.00491, equation 3.14
    couplings = rng.standard_normal(len(hyperedges))
    couplings = couplings * (1j)**(q/2) * np.sqrt((2/N)**(q-1) * J**2 * math.factorial(q-1) / q)
    majs = [dynamite.extras.majorana(i) for i in range(N)]
    return op.op_sum(op.op_product(majs[j] for j in edge) * couplings[i] for i, edge in enumerate(hyperedges))


@ut.cache('npy', 'SYK_numpy_H')
def SYK_numpy(q, rng, J=1, note=None, require_cache=False):
    return SYK(q, rng, J=J).to_numpy(sparse=False)

