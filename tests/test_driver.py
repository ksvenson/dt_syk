"""Driver checks using a NumPy Hamiltonian and a serial MPI substitute.

These test cache scheduling and archive contents, not real dynamite/MPI execution.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import utilities as ut
from tests.reference_syk import syk_components


class DriverTests(unittest.TestCase):
    def test_warm_eigensystem_skips_hamiltonian_and_globals_stay_static(self):
        class Comm:
            def Get_rank(self): return 0
            def Get_size(self): return 1
            def Gatherv(self, send, receive, root=0): receive[0][...] = send
        class Parity:
            def __init__(self, space, L): self.L = L
            def idx_to_state(self, indices):
                states = np.array([i for i in range(2**self.L) if bin(i).count('1')%2 == 0])
                return states[indices]
        dm = types.ModuleType('dynamite')
        dm.config = types.SimpleNamespace(L=4)
        spaces = types.ModuleType('dynamite.subspaces'); spaces.Parity = Parity
        tools = types.ModuleType('dynamite.tools')
        tools.MPI_COMM_WORLD = lambda: types.SimpleNamespace(tompi4py=lambda: Comm())
        factory = types.ModuleType('hamiltonian_factory')
        calls = []
        @ut.cache('npy', 'SYK42_numpy_H')
        def hamiltonian(g, rng, subspace=None, note=None):
            calls.append(g)
            parts, _ = syk_components(2*dm.config.L, rng, 'even' if subspace else 'full')
            return (1-g)*parts[0]+g*parts[1]
        factory.SYK42_numpy = hamiltonian
        default = ut.CACHE_DIR
        patches = {'dynamite': dm, 'dynamite.subspaces': spaces,
                   'dynamite.tools': tools, 'hamiltonian_factory': factory}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, patches):
            path = Path(__file__).resolve().parents[1]/'syk_temp_proj.py'
            spec = importlib.util.spec_from_file_location('driver_under_test', path)
            driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
            argv = ['syk_temp_proj.py', '--N', '8', '--g', '.5', '--realizations', '2',
                    '--tmax', '4', '--n-scr', '8', '--cache-dir', directory, '--no-plots',
                    '--gap-k-max', '3', '--gap-max-sums', '50', '--gap-lines',
                    '--thermalization-fraction', '.5', '--check-dt']
            with mock.patch.object(sys, 'argv', argv+['--g','1.1']), mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    driver.parse_args()
            with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print') as console:
                driver.main()
                self.assertEqual(console.call_count, 4)
                output = '\n'.join(str(call.args[0]) for call in console.call_args_list)
                self.assertNotIn('gap diagnostic:', output)
                self.assertNotIn('dt refinement:', output)
                self.assertIn('Analysis complete:', output)
                self.assertIn('Detailed timings and diagnostics:', output)
                archive = next(Path(directory).glob('results_*.npz'))
                with np.load(archive) as data:
                    self.assertIn('_rpenergyblocks_tol',archive.name)
                    metadata = json.loads(str(data['metadata']))
                    self.assertEqual(metadata['rp_reference'],'initial-state-adapted energy blocks')
                    first = data['temp_rp_2norm'].copy()
                    single = data['proj_single_scr_2norm'].copy()
                    gse = data['proj_instantaneous_gse_2norm'].copy()
                    self.assertEqual(single.shape,first.shape)
                    assert_allclose(single[:,:,0],0,atol=0)
                    assert_allclose(gse,0,atol=0)
                    self.assertEqual(data['moment_min_gaps'].shape, (2, 3))
                    self.assertTrue(np.isnan(data['moment_gap_times'][:, 2]).all())
                    self.assertEqual(data['temporal_threshold_times'].shape, (3,))
                    assert_allclose(data['sampling_bound_warnings'], 0)
                    self.assertGreater(float(data['analysis_wall_seconds']), 0)
                driver.main()
                with np.load(archive) as data:
                    assert_allclose(data['temp_rp_2norm'], first, atol=0)
                console.reset_mock()
                with mock.patch.object(sys, 'argv', argv+['--verbose']):
                    driver.main()
                verbose_output = '\n'.join(str(call.args[0]) for call in console.call_args_list)
                self.assertIn('gap diagnostic:', verbose_output)
                self.assertIn('dt refinement:', verbose_output)
            self.assertEqual(len(calls), 2)  # Two realizations, zero additional H builds.
            self.assertFalse(list(Path(directory).glob('SYK42_numpy_H*')))
            self.assertFalse(list(Path(directory).glob('time_evolved_overlaps*')))
            self.assertEqual(ut.CACHE_DIR, default)
            for path in Path(directory).glob('timing_*.json'):
                timing = json.loads(path.read_text())
                self.assertTrue(timing['eigensystem_cache_hit'])
                self.assertIn('gap_seconds', timing)
                self.assertIn('dt_half_max_distance_error', timing['sampling_diagnostics'])
                self.assertIn('moment_gaps', timing['sampling_diagnostics'])
                self.assertIn('warnings', timing['sampling_diagnostics'])
                basis = timing['sampling_diagnostics']['initial_state_eigenbasis']
                self.assertLess(basis['initial_state_reconstruction_error'],1e-13)
            if importlib.util.find_spec('matplotlib') is not None:
                import matplotlib
                matplotlib.use('Agg')
                import plot_N_g_sweep as plotting
                with mock.patch.object(ut, 'FIG_DIR', directory):
                    plotting.plot_archives([str(archive)], gap_lines=True, fraction=.5)
                    driver.plot_results(np.arange(5.), first, first, np.arange(1,4),
                                        'test', 'test', fraction=.5,single=single,instantaneous_gse=gse)
                self.assertEqual(len(list(Path(directory).glob('*.png'))), 9)
                # Reject mixing the old numerical-vector RP definition with
                # the adapted definition, even for different N configurations.
                with np.load(archive) as data:
                    incompatible = dict(data)
                metadata = json.loads(str(incompatible['metadata']))
                metadata['N'] = 10
                metadata.pop('rp_reference')
                incompatible['metadata'] = np.array(json.dumps(metadata))
                legacy_path = Path(directory)/'legacy_reference.npz'
                np.savez(legacy_path,**incompatible)
                with self.assertRaises(ValueError):
                    plotting.plot_archives([str(archive),str(legacy_path)])
            # End-to-end pure SYK4 case with degenerate doublets (N mod 8=4).
            # This also exercises adaptation of an eigensystem loaded from disk.
            degenerate_argv = argv+['--N','12','--g','0','--realizations','1']
            with mock.patch.object(sys,'argv',degenerate_argv), mock.patch('builtins.print'):
                driver.main()
                driver.main()
            timing_path = next(Path(directory).glob('timing_mix_N12_*.json'))
            diagnostic = json.loads(timing_path.read_text())['sampling_diagnostics']
            self.assertEqual(diagnostic['degeneracy']['block_multiplicities'],[2]*16)
            self.assertEqual(diagnostic['initial_state_eigenbasis']['occupied_energy_blocks'],16)
            assert_allclose(diagnostic['degeneracy']['degeneracy_floor_lower_bounds'],0,atol=0)
            self.assertLess(diagnostic['initial_state_eigenbasis']['ket_error_bound_at_tmax'],1e-11)


if __name__ == '__main__':
    unittest.main()
