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
            return parts[0]+g/(2*dm.config.L)*parts[1]
        factory.SYK42_numpy = hamiltonian
        default = ut.CACHE_DIR
        patches = {'dynamite': dm, 'dynamite.subspaces': spaces,
                   'dynamite.tools': tools, 'hamiltonian_factory': factory}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, patches):
            path = Path(__file__).resolve().parents[1]/'syk_temp_proj.py'
            spec = importlib.util.spec_from_file_location('driver_under_test', path)
            driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
            argv = ['syk_temp_proj.py', '--N', '8', '--g', '10', '--realizations', '2',
                    '--tmax', '4', '--n-scr', '8', '--cache-dir', directory, '--no-plots',
                    '--gap-k-max', '3', '--gap-max-sums', '50', '--gap-lines',
                    '--thermalization-fraction', '.5', '--check-dt']
            with mock.patch.object(sys, 'argv', argv), mock.patch('builtins.print'):
                driver.main()
                archive = next(Path(directory).glob('results_*.npz'))
                with np.load(archive) as data:
                    first = data['temp_rp_2norm'].copy()
                    self.assertEqual(data['moment_min_gaps'].shape, (2, 3))
                    self.assertTrue(np.isnan(data['moment_gap_times'][:, 2]).all())
                    self.assertEqual(data['temporal_threshold_times'].shape, (3,))
                driver.main()
                with np.load(archive) as data:
                    assert_allclose(data['temp_rp_2norm'], first, atol=0)
            self.assertEqual(len(calls), 2)  # Two realizations, zero additional H builds.
            self.assertFalse(list(Path(directory).glob('SYK42_numpy_H*')))
            self.assertFalse(list(Path(directory).glob('time_evolved_overlaps*')))
            self.assertEqual(ut.CACHE_DIR, default)
            for path in Path(directory).glob('timing_*.json'):
                timing = json.loads(path.read_text())
                self.assertTrue(timing['eigensystem_cache_hit'])
                self.assertIn('gap_seconds', timing)
            if importlib.util.find_spec('matplotlib') is not None:
                import matplotlib
                matplotlib.use('Agg')
                import plot_N_g_sweep as plotting
                with mock.patch.object(ut, 'FIG_DIR', directory):
                    plotting.plot_archives([str(archive)], gap_lines=True, fraction=.5)
                    driver.plot_results(np.arange(5.), first, first, np.arange(1,4),
                                        'test', 'test', fraction=.5)
                self.assertEqual(len(list(Path(directory).glob('*.png'))), 5)


if __name__ == '__main__':
    unittest.main()
