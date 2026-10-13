import itertools as it
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
import utilities as ut
import projected_funcs as pf
import thermalization as th


class CacheTests(unittest.TestCase):
    def test_explicit_directory_and_bypass(self):
        calls = []
        @ut.cache('npz', 'calculation')
        def calculation(value, note=None):
            calls.append(value)
            return {'value': np.array([value])}
        default = ut.CACHE_DIR
        with tempfile.TemporaryDirectory() as directory:
            calculation(3, note='physical', cache_dir=directory)
            assert_allclose(calculation(None, note='physical', cache_dir=directory, require_cache=True)['value'], [3])
            self.assertEqual(calls, [3])
            self.assertTrue(calculation.cache_exists(note='physical', cache_dir=directory))
            self.assertEqual(ut.CACHE_DIR, default)
            calculation(5, note='bypass', cache_dir=directory, cache_enabled=False)
            self.assertFalse(calculation.cache_exists(note='bypass', cache_dir=directory))
            with self.assertRaises(FileNotFoundError):
                calculation(None, note='missing', cache_dir=directory, require_cache=True)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_other_cache_formats(self):
        for method in ('npy', 'pkl'):
            @ut.cache(method, 'array')
            def calculation(note=None):
                return np.arange(4)
            with tempfile.TemporaryDirectory() as directory:
                assert_allclose(calculation(note='a', cache_dir=directory), np.arange(4))
                assert_allclose(calculation(note='a', cache_dir=directory), np.arange(4))

    def test_basis_cache_is_bounded_and_read_only(self):
        self.assertEqual(pf.symmetric_basis.cache_parameters()['maxsize'], 32)
        self.assertEqual(pf.symmetric_isometry.cache_parameters()['maxsize'], 8)
        tuples, factors = pf.symmetric_basis(4, 3)
        self.assertFalse(tuples.flags.writeable)
        self.assertFalse(factors.flags.writeable)
        self.assertFalse(pf.symmetric_isometry(4, 3).flags.writeable)


class TimescaleTests(unittest.TestCase):
    def test_gap_matches_all_distinct_multiset_pairs(self):
        energies = np.array([-1., .123, 2.371])
        for k in (1, 2, 3):
            sums = [sum(energies[i] for i in c) for c in it.combinations_with_replacement(range(3), k)]
            expected = min(abs(a-b) for a, b in it.combinations(sums, 2))
            result = th.minimum_moment_gap(energies, k)
            self.assertAlmostEqual(result['min_gap'], expected)
            self.assertEqual(result['status'], 'computed')
            self.assertAlmostEqual(result['reference_time'], 2*np.pi/expected)
            shifted = th.minimum_moment_gap(energies+10, k)
            self.assertAlmostEqual(shifted['min_gap'], expected)

    def test_resonance_and_guard(self):
        result = th.minimum_moment_gap(np.array([0., 1., 2.]), 2)
        self.assertEqual(result['min_gap'], 0)
        self.assertEqual(result['status'], 'zero_or_unresolved')
        self.assertIsNone(result['reference_time'])
        self.assertEqual(th.minimum_moment_gap(np.arange(20.), 3, max_sums=10)['status'], 'skipped')
        self.assertEqual(th.minimum_moment_gap(np.array([1.]), 1)['status'], 'no_pair')

    def test_threshold_including_recrossing(self):
        times = np.arange(5.)
        self.assertEqual(th.threshold_time(times, np.array([1., .05, .2, .09, .08]), .1), 3.)
        self.assertIsNone(th.threshold_time(times, np.array([1., .05, .2, .09, .11]), .1))
        self.assertIsNone(th.threshold_time(times, np.zeros(5), .1))
        self.assertEqual(th.threshold_time(times, np.array([1., .2, .15, .1, .05]), .1), 3.)
        with self.assertRaises(ValueError):
            th.threshold_time(times, np.zeros(5), 1.)


if __name__ == '__main__':
    unittest.main()
