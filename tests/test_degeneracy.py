"""Independent finite-moment checks of norms, degeneracies and partial traces."""
import itertools as it
import unittest
import numpy as np
from numpy.testing import assert_allclose
import random_phase as rp
import temporal_funcs as tf


def exact_moment(energies, populations, k, tau=None):
    """Explicit eigenbasis moment for a tiny system; None means infinite time."""
    D = len(energies)
    tuples = np.array(list(it.product(range(D), repeat=k)))
    sums = energies[tuples].sum(axis=1)
    amplitudes = np.sqrt(populations[tuples].prod(axis=1))
    gaps = sums[:, None]-sums[None, :]
    kernel = (gaps == 0) if tau is None else np.exp(-.5j*tau*gaps)*np.sinc(tau*gaps/(2*np.pi))
    return amplitudes[:, None]*amplitudes[None, :]*kernel


class DegeneracyTests(unittest.TestCase):
    def test_hierarchy_and_distinct_norm_behavior(self):
        energies = np.repeat([0.,1.,10.,100.],2)
        D = len(energies)
        pops = np.full(D, 1/D)
        diagnostic = tf.degeneracy_diagnostics(energies, pops, np.arange(1,4))
        self.assertEqual(diagnostic['block_multiplicities'], [2,2,2,2])
        hs, trace = [], []
        previous = None
        for k in (1,2,3):
            error = exact_moment(energies, pops, k)-rp.rpe_moment(pops, k)
            hs.append(np.linalg.norm(error))
            trace.append(np.linalg.norm(error, ord='nuc'))
            if previous is not None:
                reduced = np.trace(error.reshape((D,)*(2*k)), axis1=k-1, axis2=2*k-1).reshape(previous.shape)
                assert_allclose(reduced, previous, atol=1e-14)
            previous = error
        assert_allclose(hs, diagnostic['degeneracy_floor_lower_bounds'], atol=1e-14)
        self.assertTrue(np.all(np.diff(trace) >= -1e-14))
        self.assertTrue(np.all(np.diff(hs) < 0))
        self.assertTrue(np.all(np.array(hs) >= hs[0]*np.array(diagnostic['hs_partial_trace_factors'])-1e-14))

    def test_integration_agrees_with_full_moments_and_bound(self):
        energies = np.array([-.5,-.5,.73,2.1])
        pops = np.array([.1,.2,.3,.4])
        times = np.linspace(0,4,17)
        output = tf.temporal_distance_continuous(energies,pops,np.arange(1,4),times,dt=.002)
        for j,k in enumerate((1,2,3)):
            expected = [np.linalg.norm(exact_moment(energies,pops,k,t)-rp.rpe_moment(pops,k)) for t in times]
            assert_allclose(output['distance'][:,j],expected,atol=1e-9)
        diag = tf.degeneracy_diagnostics(energies,pops,np.arange(1,4))
        self.assertTrue(np.all(output['distance'] >= output['distance'][:,:1]*np.array(diag['hs_partial_trace_factors'])-1e-9))

    def test_tolerance_and_reference_basis_dependence(self):
        energies = np.array([0.,1e-12,1.])
        pops = np.array([.1,.2,.7])
        self.assertEqual(tf.degeneracy_diagnostics(energies,pops,[1],1e-14)['degenerate_blocks'],0)
        self.assertEqual(tf.degeneracy_diagnostics(energies,pops,[1],1e-11)['degenerate_blocks'],1)
        energies = np.array([0.,0.,1.])
        first = tf.degeneracy_diagnostics(energies,pops,[1])
        rotated = tf.degeneracy_diagnostics(energies,np.array([.3,0.,.7]),[1])
        assert_allclose(first['block_populations'],rotated['block_populations'])
        self.assertGreater(first['degeneracy_floor_lower_bounds'][0],0)
        self.assertAlmostEqual(rotated['degeneracy_floor_lower_bounds'][0],0)

    def test_dephased_target_preserves_degenerate_coherences(self):
        rng = np.random.default_rng(87)
        V = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
        c = rng.normal(size=4)+1j*rng.normal(size=4); c /= np.linalg.norm(c)
        energies = np.array([0.,0.,1.7,3.])
        vectors, p = tf.energy_dephased_components(energies,V,c)
        sigma = (vectors*p)@vectors.conj().T
        expected = sum(np.outer(V[:,i]*c[i],(V[:,j]*c[j]).conj())
                       for i in range(4) for j in range(4) if energies[i]==energies[j])
        assert_allclose(sigma,expected,atol=1e-14)
        rotation = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        rotatedV = V.copy(); rotatedV[:,:2] = V[:,:2]@rotation
        rotatedc = c.copy(); rotatedc[:2] = rotation.conj().T@c[:2]
        vectors2,p2 = tf.energy_dephased_components(energies,rotatedV,rotatedc)
        assert_allclose((vectors2*p2)@vectors2.conj().T,sigma,atol=1e-14)
        self.assertGreater(np.linalg.norm((V*abs(c)**2)@V.conj().T-sigma),.01)
