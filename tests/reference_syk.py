"""Independent NumPy reference for the repository's Majorana convention.

Used for scientific regression tests and bandwidth diagnostics, not production
Hamiltonian construction. chi_(2j) = Z_0 ... Z_(j-1) X_j and chi_(2j+1)
uses Y_j, matching dynamite.extras.majorana (chi^2 = I).
"""
import itertools as it
import math
import numpy as np


def syk_components(N, rng, sector='even'):
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
