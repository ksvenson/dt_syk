import importlib.util
import itertools as it
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from tests.reference_syk import syk_components


class ReferenceTests(unittest.TestCase):
    def test_against_dense_pauli_products(self):
        N, L = 8, 4
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.diag([1, -1])
        def tensor(operators):
            result = np.array([[1]])
            for operator in operators:
                result = np.kron(result, operator)
            return result
        majors = []
        # Site 0 is the least significant bit in dynamite's state mapping.
        for index in range(N):
            site = index//2
            operators = [Z if j < site else ((Y if index%2 else X) if j == site else np.eye(2))
                         for j in reversed(range(L))]
            majors.append(tensor(operators))
        rng = np.random.default_rng(91)
        components, indices = syk_components(N, np.random.default_rng(91), 'full')
        for q, actual in zip((4, 2), components):
            edges = list(it.combinations(range(N), q))
            couplings = rng.standard_normal(len(edges))*(1j)**(q/2)*np.sqrt((2/N)**(q-1)*math.factorial(q-1)/q)
            expected = np.zeros_like(actual)
            for edge, coefficient in zip(edges, couplings):
                product = np.eye(2**L, dtype=complex)
                for j in edge:
                    product = product@majors[j]
                expected += coefficient*product
            assert_allclose(actual, expected, atol=1e-14)
            assert_allclose(actual, actual.conj().T, atol=1e-14)
        compact, indices = syk_components(N, np.random.default_rng(91), 'even')
        for full, sector in zip(components, compact):
            assert_allclose(full[np.ix_(indices, indices)], sector, atol=1e-14)

    @unittest.skipUnless(importlib.util.find_spec('dynamite'), 'dynamite/PETSc not installed')
    def test_dynamite_factory_sector_and_seed(self):
        import dynamite as dm
        import dynamite.subspaces as ss
        import hamiltonian_factory as hf
        dm.config.L = 4
        seed, g, N = 91, .5, 8
        full = hf.SYK42_numpy.__wrapped__(g, np.random.default_rng(seed))
        sector = ss.Parity('even', L=4)
        indices = sector.idx_to_state(np.arange(2**3))
        compact = hf.SYK42_numpy.__wrapped__(g, np.random.default_rng(seed), subspace=sector)
        assert_allclose(compact, full[np.ix_(indices, indices)], atol=1e-14)
        parts, mapping = syk_components(N, np.random.default_rng(seed), 'even')
        assert_allclose(indices, mapping)
        assert_allclose(compact, (1-g)*parts[0]+g*parts[1], atol=1e-14)
        for slider in (0., .25, 1.):
            actual = hf.SYK42_numpy.__wrapped__(slider, np.random.default_rng(seed), subspace=sector)
            assert_allclose(actual, (1-slider)*parts[0]+slider*parts[1], atol=1e-14)
        for bad in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError):
                hf.SYK42(bad,np.random.default_rng(seed),subspace=sector)


if __name__ == '__main__':
    unittest.main()
