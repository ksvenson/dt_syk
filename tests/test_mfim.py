"""MFIM time-partition checks with an independent spin matrix and MPI substitute."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import temporal_funcs as tf
import projected_funcs as pf
import utilities as ut


def reference_mfim(L,hx=.9045,hz=.809,J=1.,periodic=True):
    """Direct computational-basis diagonal energies and single-spin flips."""
    states = np.arange(2**L)
    z = 1-2*((states[:,None]>>np.arange(L))&1)
    bonds = np.sum(z[:,:-1]*z[:,1:],axis=1)
    if periodic:bonds += z[:,-1]*z[:,0]
    H = np.diag(J*bonds+hz*z.sum(axis=1))
    for i in range(L):H[states^(1<<i),states] += hx
    return H


class ThreadWorld:
    """Exercise actual rank partition/collection code without requiring MPI here."""
    def __init__(self,size):
        self.size = size
        self.barrier = threading.Barrier(size,timeout=20)
        self.values = {}
        self.local = threading.local()

    class Comm:
        def __init__(self,world,rank):self.world,self.rank,self.step=world,rank,0
        def Get_rank(self):return self.rank
        def Get_size(self):return self.world.size
        def exchange(self,value,kind,root=0):
            w,n = self.world,self.step
            self.step += 1
            w.values[n,self.rank] = value
            w.barrier.wait()
            if self.rank==root:
                items = [w.values[n,i] for i in range(w.size)]
                if kind=='bcast':w.values[n,'answer']=items[root]
                elif kind in ('gather','allgather'):w.values[n,'answer']=items
                elif kind=='Gatherv':
                    output = items[root][1][0]
                    output[...] = np.concatenate([item[0].ravel() for item in items]).reshape(output.shape)
                    w.values[n,'answer']=None
            w.barrier.wait()
            answer = w.values[n,'answer']
            w.barrier.wait()
            return answer
        def bcast(self,value,root=0):return self.exchange(value,'bcast',root)
        def gather(self,value,root=0):
            answer=self.exchange(value,'gather',root)
            return answer if self.rank==root else None
        def allgather(self,value):return self.exchange(value,'allgather')
        def Bcast(self,array,root=0):
            value=self.bcast(array.copy() if self.rank==root else None,root)
            array[...] = value
        def Gatherv(self,send,receive,root=0):self.exchange((send.copy(),receive),'Gatherv',root)

    def run(self,function):
        def worker(rank):
            self.local.comm=self.Comm(self,rank)
            try:return function()
            except Exception:
                self.barrier.abort()
                raise
        with ThreadPoolExecutor(max_workers=self.size) as pool:
            return list(pool.map(worker,range(self.size)))


class MFIMTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'matplotlib not installed')
    def test_mfim_size_plots_have_spin_labels_and_no_g_sweep(self):
        import matplotlib
        matplotlib.use('Agg')
        import plot_N_g_sweep as plotting
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for L in (4,5):
                path=Path(directory)/f'L{L}.npz';paths.append(str(path))
                metadata={'model':'MFIM','L':L,'LA':1,'k_max':1,'n_scr':10,
                          'hx':.9045,'hz':.809,'J':1.,'boundary':'periodic'}
                data=np.array([[[1.],[.3],[.1]]])
                np.savez(path,metadata=np.array(json.dumps(metadata)),tau_series=np.arange(3.),
                    k_series=np.array([1]),temp_rp_2norm=data,proj_scr_2norm=data,
                    proj_single_scr_2norm=data,proj_single_scr_instant_2norm=data,
                    moment_gap_times=np.array([[np.nan]]),projected_sample_times=np.array([[0.,.2,1.3]]))
            with mock.patch.object(ut,'FIG_DIR',directory), \
                 mock.patch('matplotlib.figure.Figure.savefig') as save, \
                 mock.patch.object(plotting.plt,'close') as close:
                plotting.plot_archives(paths,expected_model='MFIM')
                self.assertEqual(save.call_count,5)
                for call in save.call_args_list:self.assertIn('MFIM_L_sweep',str(call.args[0]))
                figures=[call.args[0] for call in close.call_args_list]
                self.assertIn('L=4',figures[0].axes[0].get_legend_handles_labels()[1])
                self.assertNotIn('g=',figures[0].axes[0].get_title())
                assert_allclose(figures[3].axes[0].lines[0].get_xdata(),[.2,1.3])
            for figure in figures:plotting.plt.close(figure)

    def test_factory_periodic_alias_and_dense_conversion(self):
        # Load the actual factory with a small symbolic Pauli stand-in, so the
        # boundary spelling and the wraparound bond are checked without PETSc.
        dm=types.ModuleType('dynamite');dm.config=types.SimpleNamespace(L=4)
        operators=types.ModuleType('dynamite.operators')
        extras=types.ModuleType('dynamite.extras')
        boundaries=[]
        class Operator:
            def __init__(self,terms):self.terms=terms
            def __add__(self,other):return Operator(self.terms+other.terms)
            def __rmul__(self,value):return Operator([(value*c,sites) for c,sites in self.terms])
            def __mul__(self,other):
                return Operator([(c*d,sites|others)
                                 for c,sites in self.terms for d,others in other.terms])
            def to_numpy(self,sparse=False):
                result=np.zeros((2**dm.config.L,)*2)
                pauli={'X':np.array([[0.,1.],[1.,0.]]),'Z':np.diag([1.,-1.])}
                for coefficient,sites in self.terms:
                    tensor=np.array([[1.]])
                    for i in reversed(range(dm.config.L)):
                        tensor=np.kron(tensor,pauli[sites[i]] if i in sites else np.eye(2))
                    result+=coefficient*tensor
                return result
        operators.sigmax=lambda i=0:Operator([(1.,{i:'X'})])
        operators.sigmaz=lambda i=0:Operator([(1.,{i:'Z'})])
        def index_sum(operator,boundary='open'):
            self.assertIn(boundary,('open','closed'));boundaries.append(boundary)
            span=max(i for _,sites in operator.terms for i in sites)
            count=dm.config.L if boundary=='closed' else dm.config.L-span
            return Operator([(c,{(i+offset)%dm.config.L:label for i,label in sites.items()})
                             for offset in range(count) for c,sites in operator.terms])
        operators.index_sum=index_sum
        dm.extras=extras
        patches={'dynamite':dm,'dynamite.operators':operators,'dynamite.extras':extras}
        path=Path(__file__).resolve().parents[1]/'hamiltonian_factory.py'
        with mock.patch.dict(sys.modules,patches):
            spec=importlib.util.spec_from_file_location('factory_under_test',path)
            factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
            periodic=factory.MFIM_numpy.__wrapped__(boundary='periodic')
            opened=factory.MFIM().to_numpy()
        assert_allclose(periodic,reference_mfim(4),atol=1e-14)
        assert_allclose(opened,reference_mfim(4,periodic=False),atol=1e-14)
        self.assertIn('closed',boundaries)
        self.assertGreater(np.linalg.norm(periodic-opened),0)

    def test_temporal_intervals_recombine_into_serial_moments(self):
        E,p=np.array([0.,.7,1.9]),np.array([.2,.3,.5])
        times=np.linspace(0,4,9);ks=np.arange(1,4)
        serial=tf.temporal_distance_continuous(E,p,ks,times,dt=.013)
        for ranks in (3,15):
            pieces=[]
            for indices in np.array_split(np.arange(len(times)-1),ranks):
                edges=times[indices[0]:indices[-1]+2] if len(indices) else times[-1:]
                pieces.append(tf.temporal_overlap_integrals(E,p,ks,edges,.013,reporting_spacing=times[1]))
            combined=tf.temporal_distance_from_integrals(p,ks,times,
                np.concatenate([item['integral_f'] for item in pieces]),
                np.concatenate([item['integral_uf'] for item in pieces]),serial['dt_effective'])
            assert_allclose(combined['distance_squared'],serial['distance_squared'],atol=2e-14)

    def test_indexed_scrooge_samples_do_not_depend_on_partition(self):
        E,V=np.linalg.eigh(reference_mfim(3))
        c=V[0].conj();times=np.array([0.,.2,.7,1.1,2.])
        ks=np.array([1,2,3]);target={f'k{k}':np.zeros((k+1,k+1)) for k in ks}
        def calculate(ids):
            return pf.time_evolved_overlaps_and_ps_2norm.__wrapped__(E,V,c,times[ids],ks,2,target,
                instantaneous=True,n_scr=23,symmetric=True,sample_seed=91,sample_indices=ids)
        full=calculate(np.arange(len(times)))
        for ranks in (2,8):
            parts=[calculate(ids) for ids in np.array_split(np.arange(len(times)),ranks)]
            for key in full:
                assert_allclose(np.concatenate([p[key] for p in parts]),full[key],atol=1e-13)

    def test_driver_one_hamiltonian_and_identical_curves_across_rank_counts(self):
        dm=types.ModuleType('dynamite');dm.config=types.SimpleNamespace(L=4)
        tools=types.ModuleType('dynamite.tools')
        factory=types.ModuleType('hamiltonian_factory');calls=[]
        @ut.cache('npy','MFIM_numpy_H')
        def hamiltonian(hx,hz,J,boundary,note=None):
            self.assertEqual(boundary,'periodic')
            calls.append(dm.config.L)
            return reference_mfim(dm.config.L,hx,hz,J)
        factory.MFIM_numpy=hamiltonian
        patches={'dynamite':dm,'dynamite.tools':tools,'hamiltonian_factory':factory}
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(sys.modules,patches):
            path=Path(__file__).resolve().parents[1]/'mfim_temp_proj.py'
            spec=importlib.util.spec_from_file_location('mfim_driver_under_test',path)
            driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
            argv=['mfim_temp_proj.py','--L','4','--LA','1','--tmax','2','--n-scr','13',
                  '--dt','.02','--check-dt','--cache-dir',directory,'--no-plots','--gap-k-max','2']
            snapshots=[]
            for sampling in ('stratified','grid'):
                for ranks in (1,3,8):
                    world=ThreadWorld(ranks)
                    tools.MPI_COMM_WORLD=lambda:types.SimpleNamespace(tompi4py=lambda:world.local.comm)
                    with mock.patch.object(sys,'argv',argv+['--projected-sampling',sampling]),mock.patch('builtins.print'):
                        results=world.run(driver.main)
                    with np.load(results[0]) as data:
                        arrays={key:data[key].copy() for key in data.files if key.endswith('_2norm')}
                        self.assertEqual(set(arrays),{'temp_rp_2norm','proj_scr_2norm',
                            'proj_single_scr_2norm','proj_single_scr_instant_2norm'})
                        self.assertEqual(arrays['temp_rp_2norm'].shape,(1,3,3))
                        meta=json.loads(str(data['metadata']))
                        self.assertEqual(meta['model'],'MFIM');self.assertEqual(meta['boundary'],'periodic')
                        self.assertNotIn('g',meta)
                        raw=arrays['proj_single_scr_instant_2norm'][0]
                        expected=(np.concatenate((raw[:1],np.cumsum(raw[1:],axis=0)/np.arange(1,3)[:,None]))
                                  if sampling=='stratified' else np.cumsum(raw,axis=0)/np.arange(1,4)[:,None])
                        assert_allclose(arrays['proj_single_scr_2norm'][0],expected,atol=1e-14)
                        if ranks==1:snapshots.append(arrays)
                        else:
                            for key in arrays:assert_allclose(arrays[key],snapshots[-1][key],atol=2e-12)
            self.assertEqual(calls,[4])
            for path in Path(directory).glob('timing_MFIM*.json'):
                timing=json.loads(path.read_text())
                self.assertTrue(timing['eigensystem_cache_hit'])
                self.assertEqual(len(timing['rank_timings']),8)
                self.assertEqual(sum(row['states'] for row in timing['rank_timings']),3)
                self.assertEqual(sum(row['intervals'] for row in timing['rank_timings']),2)


if __name__=='__main__':unittest.main()
