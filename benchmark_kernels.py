"""Local single-thread microbenchmarks; these are not whole-job speedups.

Run: OPENBLAS_NUM_THREADS=1 python benchmark_kernels.py
No dynamite/PETSc/MPI dependency is needed.
"""
from time import perf_counter
import json
import argparse
import numpy as np
import projected_funcs as pf


def timed(function, repeats=3):
    """Measure the time required by a supplied calculation several times.

    function is called without arguments, repeats times (default three).
    Return the shortest elapsed time in seconds and the result from the last
    call. Taking the shortest time reduces interference from unrelated work;
    it is not a statistical estimate of a cluster job's completion time."""
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        samples.append(perf_counter()-start)
    return min(samples), result


def main():
    """Compare original and revised calculations on reproducible test arrays.

    Measures projected third moments, conditional density-matrix blocks,
    and reconstruction of states from energy amplitudes. Each comparison
    checks numerical agreement before reporting its time ratio. These are
    individual calculation timings, not an estimate of the complete analysis.

    If --bandwidth-N is supplied, independently construct SYK4 and SYK2 using
    explicit Majorana actions in even parity, diagonalize their combinations
    at g=0,0.25,0.5,1, and report the third-moment time-spacing bound. This uses
    the production base seed and realization zero. Results are printed as JSON.
    Use one numerical-library thread for comparison with a single MPI process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bandwidth-N', type=int, help='Also reconstruct this SYK spectrum (e.g. 24)')
    args = parser.parse_args()
    rng = np.random.default_rng(91)
    states = rng.normal(size=(8, 1024, 4))+1j*rng.normal(size=(8, 1024, 4))
    states /= np.linalg.norm(states.reshape(8, -1), axis=1)[:, None, None]
    p = np.sum(np.abs(states)**2, axis=-1, keepdims=True)
    original = lambda: np.einsum(pf.moment_constructor(3, 'projected'),
        *((states/p**(1-1/3),)*3 + (states.conj(),)*3), optimize=True).reshape(8, 64, 64)
    old_t, old = timed(original)
    new_t, compressed = timed(lambda: pf.projected_moment(states, 3))
    U = pf.symmetric_isometry(4, 3)
    np.testing.assert_allclose(U@compressed@U.T, old, atol=1e-13)
    report = {'projected_k3': {'shape': [8, 1024, 4], 'original_seconds': old_t,
              'symmetric_seconds': new_t, 'speedup': old_t/new_t}}
    D = 512
    vectors = rng.normal(size=(D, D))+1j*rng.normal(size=(D, D))
    pops = np.ones(D)/D
    def full_covariance():
        sigma = np.einsum('ab,bc', vectors, pops[:, None]*vectors.conj().T)
        return np.einsum('abac->abc', sigma.reshape(D//4, 4, D//4, 4))
    old_t, old = timed(full_covariance)
    new_t, new = timed(lambda: pf.conditional_covariances(vectors, pops, 4))
    np.testing.assert_allclose(new, old, atol=1e-13)
    report['conditional_covariance'] = {'D': D, 'original_seconds': old_t,
        'block_seconds': new_t, 'speedup': old_t/new_t}
    phases = rng.normal(size=(32, D))+1j*rng.normal(size=(32, D))
    old_t, old = timed(lambda: np.einsum('ab,cb->ca', vectors, phases))
    new_t, new = timed(lambda: phases@vectors.T)
    np.testing.assert_allclose(new, old, atol=1e-12)
    report['state_reconstruction'] = {'shape': [32, D, D], 'original_seconds': old_t,
        'blas_seconds': new_t, 'speedup': old_t/new_t}
    if args.bandwidth_N is not None:
        from scipy.linalg import eigvalsh
        from tests.reference_syk import syk_components
        parts, _ = syk_components(args.bandwidth_N, np.random.default_rng(
            np.random.SeedSequence(628**3, spawn_key=(0, 0))))
        report['bandwidth_reference'] = []
        for g in (0, .25, .5, 1):
            values = eigvalsh((1-g)*parts[0]+g*parts[1])
            W = float(np.ptp(values))
            report['bandwidth_reference'].append({'N': args.bandwidth_N, 'g': g,
                'bandwidth': W, 'k3_nyquist_dt': np.pi/(3*W)})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
