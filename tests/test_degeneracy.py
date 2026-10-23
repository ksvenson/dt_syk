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
    def test_initial_state_basis_preserves_hamiltonian_and_evolution(self):
        rng = np.random.default_rng(124)
        V = np.linalg.qr(rng.normal(size=(7,7))+1j*rng.normal(size=(7,7)))[0]
        energies = np.array([2.,0.,2.,0.,0.,5.,5.])
        c = np.array([.2j,.3,.4,.1j,.5,0.,0.], complex)
        c /= np.linalg.norm(c)
        state = V@c
        original = V.copy()
        output = tf.initial_state_eigenbasis(energies,V,state,0)
        W, a = output['evecs'], output['coefficients']
        assert_allclose(V,original,atol=0)
        assert_allclose(output['evals'],energies,atol=0)
        assert_allclose(W.conj().T@W,np.eye(7),atol=2e-15)
        assert_allclose(W@a,state,atol=2e-15)
        assert_allclose(W.conj().T@state,a,atol=2e-15)
        assert_allclose((W*energies)@W.conj().T,(V*energies)@V.conj().T,atol=5e-15)
        for energy in np.unique(energies):
            indices = np.flatnonzero(energies==energy)
            # The exactly unoccupied block is handled in the separate edge test.
            first = indices[0]
            assert_allclose(a[indices[1:]],0,atol=0)
            if abs(a[first]) > 1e-14:
                component = V[:,indices]@c[indices]
                assert_allclose(W[:,first],component/np.linalg.norm(component),atol=2e-15)
        for time in (0.,.2,31.,1e5):
            expected = V@(np.exp(-1j*time*energies)*c)
            actual = W@(np.exp(-1j*time*output['evals'])*a)
            assert_allclose(actual,expected,atol=3e-15)
        # Independent full-matrix time dephasing equals the new RP first moment.
        raw = np.outer(c,c.conj())*(energies[:,None]==energies[None,:])
        sigma = V@raw@V.conj().T
        assert_allclose((W*abs(a)**2)@W.conj().T,sigma,atol=2e-15)

    def test_adapted_reference_invariant_under_degenerate_rotations(self):
        rng = np.random.default_rng(125)
        V = np.linalg.qr(rng.normal(size=(5,5))+1j*rng.normal(size=(5,5)))[0]
        state = rng.normal(size=5)+1j*rng.normal(size=5)
        state /= np.linalg.norm(state)
        energies = np.array([0.,0.,0.,1.,1.])
        rotated = V.copy()
        for indices in ([0,1,2],[3,4]):
            m = len(indices)
            U = np.linalg.qr(rng.normal(size=(m,m))+1j*rng.normal(size=(m,m)))[0]
            rotated[:,indices] = V[:,indices]@U
        first = tf.initial_state_eigenbasis(energies,V,state)
        second = tf.initial_state_eigenbasis(energies,rotated,state)
        for result in (first,second):
            assert_allclose(result['coefficients'][[1,2,4]],0,atol=0)
        assert_allclose(first['evecs'][:,[0,3]],second['evecs'][:,[0,3]],atol=2e-15)
        assert_allclose(first['coefficients'],second['coefficients'],atol=2e-15)
        # Exact moment check at k=1,2,3 for two distinct occupied energies.
        for k in (1,2,3):
            active = first['coefficients'] > 0
            p = abs(first['coefficients'][active])**2
            infinite = exact_moment(first['evals'][active],p,k)
            assert_allclose(infinite,rp.rpe_moment(p,k),atol=2e-15)

    def test_adaptation_edge_cases_and_numerical_grouping_bound(self):
        # Includes an empty degenerate block and a first component equal to -1:
        # the reflection must stay finite even for an already aligned direction.
        energies = np.array([0.,0.,1.,1.])
        state = np.array([-1.,0.,0.,0.])
        output = tf.initial_state_eigenbasis(energies,np.eye(4),state)
        assert_allclose(output['evecs']@output['coefficients'],state,atol=0)
        assert_allclose(output['evecs'][:,2:],np.eye(4)[:,2:],atol=0)
        self.assertEqual(np.count_nonzero(output['coefficients']),1)
        with self.assertRaises(ValueError):
            tf.initial_state_eigenbasis(energies,np.eye(4),state*2)
        # A deliberately resolvable small splitting: grouping is an explicit
        # approximation, and tolerance=0 must preserve that physical splitting.
        energies = np.array([0.,1e-7,1.])
        state = np.sqrt([.2,.3,.5])
        grouped = tf.initial_state_eigenbasis(energies,np.eye(3),state,1e-6)
        separate = tf.initial_state_eigenbasis(energies,np.eye(3),state,0)
        assert_allclose(separate['evals'],energies,atol=0)
        self.assertEqual(np.count_nonzero(separate['coefficients']),3)
        self.assertEqual(np.count_nonzero(grouped['coefficients']),2)
        delta = grouped['basis_diagnostics']['max_energy_shift']
        self.assertGreater(delta,0)
        for time in (1.,1e4):
            before = np.exp(-1j*time*energies)*state
            after = grouped['evecs']@(np.exp(-1j*time*grouped['evals'])*grouped['coefficients'])
            self.assertLessEqual(np.linalg.norm(after-before),time*delta+2e-15)
        # Adjacent blocks must not be merged again after assigning their means.
        grouped = tf.initial_state_eigenbasis(np.array([0.,.9,1.1]),np.eye(3),state,1.)
        diag = tf.degeneracy_diagnostics(grouped['evals'],abs(grouped['coefficients'])**2,[1],0)
        self.assertEqual(diag['block_multiplicities'],[2,1])

    def test_adapted_first_moment_convergence_and_higher_resonance(self):
        energies = np.repeat([0.,1.,2.],2)
        state = np.sqrt(np.array([.1,.2,.1,.2,.2,.2]))
        result = tf.initial_state_eigenbasis(energies,np.eye(6),state,0)
        p = abs(result['coefficients'])**2
        diag = tf.degeneracy_diagnostics(result['evals'],p,[1,2,3],0)
        assert_allclose(diag['degeneracy_floor_lower_bounds'],0,atol=0)
        active = p>0
        E,p = result['evals'][active],p[active]
        assert_allclose(exact_moment(E,p,1),rp.rpe_moment(p,1),atol=1e-15)
        # 0+2=1+1 remains a genuine second-order resonance after adaptation.
        self.assertGreater(np.linalg.norm(exact_moment(E,p,2)-rp.rpe_moment(p,2)),.01)
        times = np.linspace(0,2*np.pi,9)
        computed = tf.temporal_distance_continuous(result['evals'],abs(result['coefficients'])**2,
                                                   [1,2],times,dt=.002)
        for j,k in enumerate((1,2)):
            expected = [np.linalg.norm(exact_moment(E,p,k,t)-rp.rpe_moment(p,k)) for t in times]
            assert_allclose(computed['distance_squared'][:,j],np.square(expected),atol=2e-12)
        self.assertLess(computed['distance'][-1,0],1e-6)

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
