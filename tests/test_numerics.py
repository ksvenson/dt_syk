import itertools as it
import math
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

import projected_funcs as pf
import temporal_funcs as tf
import random_phase as rp


def tensor_reference(states, k):
    """Independent full tensor formula with explicit Born probabilities."""
    output = np.zeros(states.shape[:-2]+(states.shape[-1]**k,)*2, dtype=complex)
    for prefix in np.ndindex(states.shape[:-2]):
        for state in states[prefix]:
            p = np.vdot(state, state).real
            if p == 0:
                continue
            vector = state/np.sqrt(p)
            tensor = vector
            for _ in range(k-1):
                tensor = np.kron(tensor, vector)
            output[prefix] += p*np.outer(tensor, tensor.conj())
    return output


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)

    def states(self, shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)

    def test_symmetric_moments_and_distances(self):
        states = self.states((3, 5, 4))
        states[:, 0] = 0
        states /= np.linalg.norm(states.reshape(3, -1), axis=1)[:, None, None]
        for k in (1, 2, 3):
            reference = tensor_reference(states, k)
            compressed = pf.projected_moment(states, k)
            U = pf.symmetric_isometry(4, k)
            assert_allclose(U.T@U, np.eye(U.shape[1]), atol=1e-14)
            assert_allclose(U@compressed@U.T, reference, atol=1e-14)
            assert_allclose(np.trace(compressed, axis1=-2, axis2=-1), 1, atol=1e-14)
            assert_allclose(np.linalg.norm(compressed[0]-compressed[1]),
                            np.linalg.norm(reference[0]-reference[1]), atol=1e-14)
        # A single occupied outcome must be finite even at k>1.
        initial = np.zeros((1, 4, 4), complex)
        initial[0, 0, 0] = 1
        assert_allclose(pf.projected_moment(initial, 3)[0, 0, 0], 1)

    def test_covariance_blocks_and_sector_mapping(self):
        indices = np.array([i for i in range(16) if bin(i).count("1")%2 == 0])[::-1]
        vectors = np.linalg.qr(self.states((8, 8)))[0]
        pops = self.rng.dirichlet(np.ones(8))
        lifted = np.zeros((16, 8), complex)
        lifted[indices] = vectors
        sigma = (lifted*pops)@lifted.conj().T
        expected = np.stack([sigma[z:z+4, z:z+4] for z in range(0, 16, 4)])
        for chunk in (1, 3, 32):
            actual = pf.conditional_covariances(vectors, pops, 4, indices, 16, chunk)
            assert_allclose(actual, expected, atol=1e-14)
        assert_allclose(pf.conditional_covariances(lifted, pops, 4), expected, atol=1e-14)

    def test_scrooge_against_full_tensor_formula(self):
        vectors = np.linalg.qr(self.states((8, 8)))[0]
        pops = self.rng.dirichlet(np.ones(8))
        seed, n, DA = 31, 17, 4
        cov = pf.conditional_covariances(vectors, pops, DA)
        vals, vecs = np.linalg.eigh(cov)
        roots = (vecs*np.sqrt(vals)[:, None, :])@vecs.conj().swapaxes(-1, -2)
        rng = np.random.default_rng(seed)
        haar = rng.normal(size=(n, DA))+1j*rng.normal(size=(n, DA))
        haar /= np.linalg.norm(haar, axis=1)[:, None]
        states = haar@roots.swapaxes(-1, -2)
        output = pf.cond_scr_moment.__wrapped__(vectors, pops, n, np.arange(1, 4), DA,
                     np.random.default_rng(seed), symmetric=False, bath_chunk=1)
        assert_allclose(output['k1'], cov.sum(axis=0), atol=1e-14)
        for k in (2, 3):
            assert_allclose(output[f'k{k}'], DA/n*tensor_reference(states, k).sum(axis=0), atol=1e-14)
        compressed = pf.cond_scr_moment.__wrapped__(vectors, pops, n, np.arange(1, 4), DA,
                     np.random.default_rng(seed), symmetric=True, bath_chunk=32)
        for k in (1, 2, 3):
            U = pf.symmetric_isometry(DA, k)
            assert_allclose(U@compressed[f'k{k}']@U.T, output[f'k{k}'], atol=1e-14)

    def test_projected_evolution_full_and_parity(self):
        indices = np.array([0, 3, 5, 6])
        vectors = np.linalg.qr(self.states((4, 4)))[0]
        evals = np.array([-3., -1., .2, 2.])
        coefficients = vectors[0].conj()
        times = np.array([0., .2, .5, 3.])
        target = {f'k{k}': np.zeros((2**k,)*2) for k in (1, 2, 3)}
        output = pf.time_evolved_overlaps_and_ps_2norm.__wrapped__(evals, vectors,
                coefficients, times, np.arange(1, 4), 2, target,
                chunk=3, basis_indices=indices, full_dimension=8)
        compact = (np.exp(-1j*times[:, None]*evals)*coefficients)@vectors.T
        full = np.zeros((4, 8), complex)
        full[:, indices] = compact
        for j, k in enumerate((1, 2, 3)):
            expected = tensor_reference(full.reshape(4, 4, 2), k)
            assert_allclose(output['2norm'][:, j], np.linalg.norm(expected, axis=(-2, -1)), atol=1e-14)
        assert_allclose(output['overlaps'], np.exp(-1j*times[:, None]*evals)@np.abs(coefficients)**2, atol=1e-14)
        lifted = np.zeros((8, 4), complex)
        lifted[indices] = vectors
        full_output = pf.time_evolved_overlaps_and_ps_2norm.__wrapped__(evals, lifted,
                coefficients, times, np.arange(1, 4), 2, target, chunk=1)
        assert_allclose(output['2norm'], full_output['2norm'], atol=1e-14)

    def test_continuous_temporal_against_exact_multiset_sums(self):
        evals = np.array([-1., .2, 2.3])
        pops = np.array([.2, .3, .5])
        times = np.linspace(0, 8, 41)
        baseline = rp.rpe_square_2norm(pops, 3)
        exact = np.empty((len(times), 3))
        for j, k in enumerate((1, 2, 3)):
            ordered = np.array(list(it.product(range(3), repeat=k)))
            energies = evals[ordered].sum(axis=1)
            weights = pops[ordered].prod(axis=1)
            gaps = energies[:, None]-energies[None, :]
            for i, t in enumerate(times):
                purity = np.sum(weights[:, None]*weights[None, :]*np.sinc(gaps*t/(2*np.pi))**2)
                exact[i, j] = purity-baseline[j]
        output = tf.temporal_distance_continuous(evals, pops, np.arange(1, 4), times, dt=.005, chunk=37)
        assert_allclose(output['distance_squared'], exact, atol=8e-6, rtol=0)
        refined = tf.temporal_distance_continuous(evals, pops, np.arange(1, 4), times, dt=.0025, chunk=131)
        self.assertLess(np.max(np.abs(refined['distance_squared']-exact)),
                        .3*np.max(np.abs(output['distance_squared']-exact)))
        shifted = tf.temporal_distance_continuous(evals+10, pops, np.arange(1, 4), times, dt=.005, chunk=113)
        assert_allclose(shifted['distance_squared'], output['distance_squared'], atol=1e-13)

    def test_resonances_leave_nonzero_distance(self):
        # All energy eigenvectors share one eigenvalue; the orbit is constant.
        output = tf.temporal_distance_continuous(np.ones(3), np.ones(3)/3,
                    np.arange(1, 4), np.arange(5.), dt=.1)
        expected = 1-rp.rpe_square_2norm(np.ones(3)/3, 3)
        assert_allclose(output['distance_squared'], np.tile(expected, (5, 1)), atol=1e-13)

    def test_random_phase_recurrence(self):
        pops = np.array([.1, .3, .6])
        for k in range(1, 5):
            moment = rp.rpe_moment(pops, k)
            assert_allclose(rp.rpe_square_2norm(pops, k)[-1], np.sum(moment**2), atol=1e-14)

    def test_energy_histogram_matches_pair_enumeration(self):
        for evals in (np.array([0., .1, .2, .4, .5]), np.array([-2., -2., -1., 1., 3.])):
            for k in (1, 2, 3):
                sums = np.array([evals[list(c)].sum() for c in it.combinations(range(len(evals)), k)])
                diffs = np.array([abs(a-b) for a, b in it.combinations(sums, 2)])
                for edges in (np.array([0., .1, .2, .3, 1.]), np.array([.1, .2, .4, 10.]), np.array([-1., 0., .1, 10.])):
                    result = tf.eng_diffs.__wrapped__(evals, k, edges, chunk=3, verbose=False)
                    assert_allclose(result['counts'], np.histogram(diffs, bins=edges)[0], atol=0)
                    self.assertEqual(result['min'], diffs.min())
        # A lone subset has no nontrivial pair.
        result = tf.eng_diffs.__wrapped__(np.arange(3.), 3, np.array([0., 4.]), verbose=False)
        self.assertTrue(np.isinf(result['min']))
        self.assertEqual(result['counts'][0], 0)

    def test_exact_temporal_legacy_formula_and_guard(self):
        evals = np.array([-1., 0., 2.])
        pops = np.array([.2, .3, .5])
        times = np.array([0., .3, 2.])
        for k in (1, 2, 3):
            ordered = np.array(list(it.product(range(3), repeat=k)))
            energies = evals[ordered].sum(axis=1)
            weights = pops[ordered].prod(axis=1)
            gaps = energies[:, None]-energies[None, :]
            expected = np.array([np.sum(weights[:, None]*weights[None, :]*
                np.sinc(gaps*t/(2*np.pi))**2) for t in times])-rp.rpe_square_2norm(pops, k)[-1]
            actual = tf.temp_square_2norm_exact.__wrapped__(evals, pops, k, times, chunk=3, verbose=False)
            assert_allclose(actual, expected, atol=1e-14)
        with self.assertRaises(ValueError):
            tf.temp_square_2norm_exact.__wrapped__(evals, pops, 3, times, max_sums=1)

    def test_bandwidth_diagnostic(self):
        diag = tf.spectral_sampling_diagnostics(np.array([-2., 3.]), 3, 1.)
        self.assertFalse(diag['satisfies_bound'])
        self.assertAlmostEqual(diag['nyquist_dt'], np.pi/15)


if __name__ == '__main__':
    unittest.main()
