import numpy as np
import math
import dynamite as dm
import dynamite.operators as op
import dynamite.extras
import itertools as it

import utilities as ut


def MFIM(hx=0.8090, hy=0.9045, J=1, boundary='open'):
    """Mixed-field Ising operator H=sum_i [J X_i X_(i+1)+hx X_i+hy Y_i].

    Set dynamite.config.L to the number of spin-1/2 sites before calling.
    boundary='periodic' is an alias for Dynamite's 'closed' convention;
    'open' omits the bond between the last and first sites.

    The defaults are the model of Kim, Ikeda and Huse,
    Phys. Rev. E 90, 052105 (2014), expressed in rotated spin axes. The
    same on-site unitary u=(I-i*(X+Y+Z))/2 sends X to Y, Y to Z, and Z to X.
    It maps the previous J ZZ+0.9045 X+0.8090 Z Hamiltonian to this one,
    preserving the spectrum. This function rotates only the Hamiltonian:
    it does not rotate an initial ket or change a measurement basis.
    In particular, the analysis still prepares |0...0>_Z and measures B in Z.

    X and Y have zero diagonal in the Z basis, so every Z product ket has
    zero mean energy. With a nonzero Y field the matrix is generally complex
    Hermitian; its imaginary entries are physical and must be retained.
    """
    if boundary == 'periodic':
        boundary = 'closed'
    if boundary not in ('open','closed'):
        raise ValueError('boundary must be open, closed or periodic')
    # separate field and interaction terms so that boundary conditions can be applied
    # properly.
    field = op.index_sum(hx * op.sigmax() + hy * op.sigmay())
    interaction = op.index_sum(J * op.sigmax(i=0) * op.sigmax(i=1), boundary=boundary)
    return field + interaction


@ut.cache('npy', 'MFIM_XX_XY_numpy_H')
def MFIM_numpy(hx=0.8090, hy=0.9045, J=1, boundary='periodic', note=None):
    """Dense matrix of the XX Ising operator with X and Y fields.

    dynamite.config.L counts spin-1/2 sites. The new MFIM analysis explicitly
    uses periodic boundaries and the rotated published chaotic parameters of
    Kim, Ikeda and Huse, Phys. Rev. E 90, 052105 (2014), Eq. (7):
    H=sum_i [J X_i X_(i+1)+hx X_i+hy Y_i]. MFIM's historical default remains
    open; this wrapper defaults to the periodic model requested by the driver.
    The returned matrix uses the unchanged computational Z basis. Its generally
    complex entries are preserved for the Hermitian eigensolver. The cache base
    identifies these rotated axes, including for calls without a custom note.
    No SYK fermion-parity restriction is applied: the fields mix Z-parity sectors.
    """
    if boundary not in ('open','closed','periodic'):
        raise ValueError('boundary must be open, closed or periodic')
    if not np.all(np.isfinite([hx,hy,J])):
        raise ValueError('MFIM couplings must be finite')
    return MFIM(hx=hx,hy=hy,J=J,boundary=boundary).to_numpy(sparse=False)


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


