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
    boundary='periodic' is an alias for dynamite's 'closed' convention.
    """
    if boundary == 'periodic':
        boundary = 'closed'
    if boundary not in ('open','closed'):
        raise ValueError('boundary must be open, closed or periodic')
    # separate field and interaction terms so that boundary conditions can be applied
    # properly.
    field = op.index_sum(hx * op.sigmax() + hz * op.sigmaz())
    interaction = op.index_sum(J * op.sigmaz(i=0) * op.sigmaz(i=1), boundary=boundary)
    return field + interaction


@ut.cache('npy', 'MFIM_numpy_H')
def MFIM_numpy(hx=0.9045, hz=0.8090, J=1, boundary='periodic', note=None):
    """Dense matrix of the existing mixed-field Ising operator.

    dynamite.config.L counts spin-1/2 sites. The new MFIM analysis explicitly
    uses periodic boundaries, with the published chaotic parameters of
    Kim, Ikeda and Huse, Phys. Rev. E 90, 052105 (2014), Eq. (7).
    This conversion does not change signs, field strengths or Pauli conventions:
    H=sum_i [J Z_i Z_(i+1)+hx X_i+hz Z_i]. MFIM's historical default remains
    open; this wrapper defaults to the periodic model requested by the driver.
    No SYK fermion-parity restriction is applied: X_i mixes Z-parity sectors.
    """
    if boundary not in ('open','closed','periodic'):
        raise ValueError('boundary must be open, closed or periodic')
    if not np.all(np.isfinite([hx,hz,J])):
        raise ValueError('MFIM couplings must be finite')
    # This Pauli Hamiltonian is real for real fields; use the real symmetric
    # eigensolver rather than retaining an unnecessary complex zero-imaginary part.
    return np.real_if_close(MFIM(hx=hx,hz=hz,J=J,boundary=boundary).to_numpy(sparse=False))


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


