"""Independent checks of the two instantaneous Scrooge targets."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
import projected_funcs as pf
from tests.test_numerics import tensor_reference


class InstantaneousTests(unittest.TestCase):
    def test_single_scrooge_matches_probability_adjusted_formula(self):
        rng = np.random.default_rng(17)
        U = np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
        rho = (U*np.array([0.,.25,.75]))@U.conj().T
        n,seed = 23,92
        actual = pf.scrooge_moments(rho,n,[1,2,3],np.random.default_rng(seed),symmetric=False)
        values,vectors = np.linalg.eigh(rho)
        keep = values > 64*np.finfo(float).eps
        g = np.random.default_rng(seed).standard_normal((1,n,3,2))
        h = (g[...,0]+1j*g[...,1])*keep[None,None,:]
        h /= np.linalg.norm(h,axis=-1,keepdims=True)
        states = (h*np.sqrt(np.maximum(values,0))[None,None,:])@vectors.T
        assert_allclose(actual['k1'],rho,atol=1e-14)
        for k in (2,3):
            assert_allclose(actual[f'k{k}'],2/n*tensor_reference(states,k)[0],atol=1e-13)

    def test_pure_full_generalized_target_is_projected_moment(self):
        rng = np.random.default_rng(8)
        states = rng.normal(size=(3,5,4))+1j*rng.normal(size=(3,5,4))
        states /= np.linalg.norm(states.reshape(3,-1),axis=1)[:,None,None]
        states[:,0] = 0  # zero-probability outcome is also supported
        states /= np.linalg.norm(states.reshape(3,-1),axis=1)[:,None,None]
        blocks = states[..., :,None]*states[...,None,:].conj()
        targets = pf.scrooge_moments(blocks,11,[1,2,3],np.random.default_rng(9),symmetric=False)
        pure_gse = pf.generalized_scrooge_pure(states,[1,2,3],symmetric=False)
        for k in (1,2,3):
            expected = tensor_reference(states,k)
            assert_allclose(targets[f'k{k}'].sum(axis=1),expected,atol=1e-13)
            assert_allclose(pure_gse[f'k{k}'],expected,atol=1e-13)

    def test_bell_state_single_target_has_parity_obstruction(self):
        states = np.eye(2,dtype=complex)/np.sqrt(2)
        projected = pf.projected_moment(states,2)
        single = pf.scrooge_moments(np.eye(2)/2,40000,[1,2],np.random.default_rng(71))
        assert_allclose(single['k1'],np.eye(2)/2,atol=0)
        assert_allclose(single['k2'],np.eye(3)/3,atol=.004)
        self.assertAlmostEqual(np.linalg.norm(projected-single['k2']),np.sqrt(1/6),delta=.004)

    def test_evolution_batch_independence_and_analytic_zeros(self):
        rng = np.random.default_rng(61)
        V = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
        energies = np.array([-1.,.2,.9,2.])
        times = np.array([0.,.1,.6,2.,5.])
        ks = np.arange(1,4)
        target = {f'k{k}':np.zeros((k+1,k+1),complex) for k in ks}
        outputs = []
        for chunk in (1,3):
            outputs.append(pf.time_evolved_overlaps_and_ps_2norm.__wrapped__(
                energies,V,V[0].conj(),times,ks,2,target,chunk=chunk,symmetric=True,
                instantaneous=True,n_scr=37,rng=np.random.default_rng(99)))
        for key in outputs[0]:
            assert_allclose(outputs[0][key],outputs[1][key],atol=1e-13)
        assert_allclose(outputs[0]['single_scr_2norm'][:,0],0,atol=0)
        assert_allclose(outputs[0]['instantaneous_gse_2norm'],0,atol=0)
        assert_allclose(outputs[0]['single_scr_2norm'][0],0,atol=1e-13)

    def test_invalid_density_matrix(self):
        for rho in (np.diag([1.,-.1]),np.array([[1.,1.],[0.,1.]])):
            with self.assertRaises(ValueError):
                pf.scrooge_moments(rho,10,[2],np.random.default_rng(1))
