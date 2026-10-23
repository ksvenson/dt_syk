"""Independent NumPy reference for the repository's Majorana convention.

Used for scientific regression tests and bandwidth diagnostics, not production
Hamiltonian construction. chi_(2j) = Z_0 ... Z_(j-1) X_j and chi_(2j+1)
uses Y_j, matching dynamite.extras.majorana (chi^2 = I).
"""
import itertools as it
import math
import numpy as np


def syk_components(N, rng, sector='even'):
    """Independently construct SYK4 and SYK2 matrices for verification.

    Parameters
    ----------
    N : even integer, at least four
        Number of Majorana operators. There are N/2 two-level sites.
    rng : numpy.random.Generator
        Supplies real independent Gaussian couplings, in increasing-order
        combinations of Majorana indices. SYK4 couplings are drawn first,
        followed by SYK2, matching the production Hamiltonian factory.
    sector : 'even' or 'full', default 'even'
        Restrict to computational states with an even number of occupied
        sites, or retain the whole Hilbert space.

    Returns
    -------
    matrices : list of two complex arrays
        [H4,H2], before applying the slider weights. For q=4 or 2 the coefficient of
        chi_i1...chi_iq is i**(q/2)*sqrt((2/N)**(q-1)*(q-1)!/q)*J_edge,
        with J_edge standard normal. Each Majorana squares to identity:
        chi_2j=Z_0...Z_(j-1)X_j and chi_(2j+1)=Z_0...Z_(j-1)Y_j.
    indices : integer array
        Full computational-basis labels corresponding to matrix rows/columns;
        site zero is the least significant binary digit.

    This reference applies the rightmost Majorana first to every retained
    basis ket and adds its signed amplitude to the Hamiltonian. It needs no
    dynamite installation and is intended for tests and spectral diagnostics.
    Matrices are dense, so large N can require substantial storage."""
    if N < 4 or N % 2 or sector not in ('even', 'full'):
        raise ValueError('Need even N and sector even/full')
    D = 2**(N//2)
    parity = np.array([bin(i).count('1')%2 for i in range(D)])
    indices = np.flatnonzero(parity == 0) if sector == 'even' else np.arange(D)
    inverse = np.full(D, -1)
    inverse[indices] = np.arange(len(indices))
    columns = np.arange(len(indices))
    matrices = []
    for q in (4, 2):
        edges = list(it.combinations(range(N), q))
        couplings = rng.standard_normal(len(edges))*(1j)**(q/2)*np.sqrt(
            (2/N)**(q-1)*math.factorial(q-1)/q)
        H = np.zeros((len(indices),)*2, complex)
        for edge, coefficient in zip(edges, couplings):
            states = indices.copy()
            amplitude = np.ones(len(states), complex)
            for majorana in reversed(edge):
                site = majorana//2
                amplitude *= np.where(parity[states & ((1<<site)-1)], -1, 1)
                if majorana%2:
                    amplitude *= 1j*np.where((states>>site)&1, -1, 1)
                states ^= 1<<site
            H[inverse[states], columns] += coefficient*amplitude
        matrices.append(H)
    return matrices, indices
