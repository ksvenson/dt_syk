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


def SYK(q, rng, J=1, subspace=None):
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
    ret = op.op_sum(op.op_product(majs[j] for j in edge) * couplings[i] for i, edge in enumerate(hyperedges))
    if subspace is not None:
        ret.add_subspace(subspace)
    return ret


@ut.cache('npy', 'SYK_numpy_H')
def SYK_numpy(q, rng, J=1, subspace=None, note=None, require_cache=False):
    spaces = None if subspace is None else (subspace, subspace)
    return SYK(q, rng, J=J, subspace=subspace).to_numpy(subspaces=spaces, sparse=False)


def SYK42(g, rng, J=1, subspace=None):
    """Interpolate between SYK4 and SYK2 as (1-g)*H4+g*H2, 0<=g<=1.

    Each component retains SYK's existing q-dependent random-coupling
    normalization and common J scale. g is dimensionless; there is no extra
    1/N factor. Both components are drawn in the fixed order H4 then H2 so
    the same seed gives the same component Hamiltonians across the slider.
    The total spectral width need not stay constant as g changes.
    """
    if not np.isfinite(g) or not 0 <= g <= 1:
        raise ValueError('g must be a finite slider value between zero and one')
    return (1-g)*SYK(4, rng, J=J, subspace=subspace) + g*SYK(2, rng, J=J, subspace=subspace)


@ut.cache('npy', 'SYK42_numpy_H')
def SYK42_numpy(g, rng, J=1, subspace=None, note=None, require_cache=False):
    spaces = None if subspace is None else (subspace, subspace)
    return SYK42(g, rng, J=J, subspace=subspace).to_numpy(subspaces=spaces, sparse=False)


