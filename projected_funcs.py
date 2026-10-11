"""Projected/conditional Scrooge moments, with basis convention psi.reshape(-1, DA).

The optional symmetric representation is orthonormal, so Frobenius distances
are identical to those in the full tensor space. Zero-probability outcomes
contribute zero, including at t=0 and in a fixed parity sector.
"""
import itertools as it
import math
import string
from functools import lru_cache

import numpy as np

import utilities as ut


def moment_constructor(k, ensemble):
    """Legacy tensor contraction, retained for reference and callers."""
    letters = string.ascii_lowercase
    if k < 1 or 2*k+2 >= len(letters):
        raise ValueError('Unsupported moment order')
    inputs = ','.join('ab' + i for i in letters[2:2*k+2])
    output = ''.join(letters[2:2*k+2])
    if ensemble == 'projected':
        return inputs + '->a' + output
    if ensemble == 'scrooge':
        return inputs + '->' + output
    raise ValueError('Unknown ensemble: ' + ensemble)


@lru_cache(None)
def symmetric_basis(DA, k):
    """Occupation tuples and sqrt(k! / prod occupation!) in a fixed order."""
    if DA < 1 or k < 1:
        raise ValueError('DA and k must be positive')
    tuples = np.array(list(it.combinations_with_replacement(range(DA), k)))
    factors = np.array([
        math.sqrt(math.factorial(k) / math.prod(
            math.factorial(c) for c in np.bincount(t, minlength=DA)))
        for t in tuples
    ])
    return tuples, factors


@lru_cache(None)
def symmetric_isometry(DA, k):
    """Columns embed orthonormal symmetric coordinates into tensor coordinates."""
    tuples, factors = symmetric_basis(DA, k)
    embedding = np.zeros((DA**k, len(tuples)))
    for j, t in enumerate(tuples):
        for p in set(it.permutations(t)):
            embedding[np.ravel_multi_index(p, (DA,)*k), j] = 1/factors[j]
    return embedding


def weighted_features(states, k):
    """sqrt(p) * |normalized state>^tensor k, in symmetric coordinates."""
    DA = states.shape[-1]
    tuples, factors = symmetric_basis(DA, k)
    root_p = np.sqrt(np.sum(np.abs(states)**2, axis=-1))
    unit = np.zeros_like(states)
    np.divide(states, root_p[..., None], out=unit, where=root_p[..., None] > 0)
    features = np.ones(states.shape[:-1] + (len(tuples),), dtype=states.dtype)
    for j in range(k):
        features *= unit[..., tuples[:, j]]
    features *= factors * root_p[..., None]
    return features


def projected_moment(states, k, symmetric=True):
    """Sum outcome-weighted moments; states has shape (..., outcomes, DA)."""
    features = weighted_features(states, k)
    moment = features.swapaxes(-1, -2) @ features.conj()
    if not symmetric:
        embedding = symmetric_isometry(states.shape[-1], k)
        moment = embedding @ moment @ embedding.T
    return moment


def _basis_map(evecs, DA, basis_indices, full_dimension):
    D = evecs.shape[0] if full_dimension is None else int(full_dimension)
    if D % DA:
        raise ValueError('full_dimension must be divisible by DA')
    if basis_indices is None:
        basis_indices = np.arange(D)
    indices = np.asarray(basis_indices, dtype=np.int64)
    if (indices.size != evecs.shape[0] or np.unique(indices).size != indices.size
            or np.any(indices < 0) or np.any(indices >= D)):
        raise ValueError('Invalid computational basis mapping')
    inverse = np.full(D, -1, dtype=np.int64)
    inverse[indices] = np.arange(indices.size)
    return D, indices, inverse


def conditional_covariances(evecs, pops, DA, basis_indices=None,
                            full_dimension=None, bath_chunk=32):
    """Compute only sigma_{A|z} blocks, without constructing the full sigma."""
    D, _, inverse = _basis_map(evecs, DA, basis_indices, full_dimension)
    if bath_chunk < 1:
        raise ValueError('bath_chunk must be positive')
    DB = D // DA
    result = np.empty((DB, DA, DA), dtype=np.complex128)
    for start in range(0, DB, bath_chunk):
        stop = min(start+bath_chunk, DB)
        mapping = inverse[start*DA:stop*DA].reshape(-1, DA)
        rows = np.zeros(mapping.shape + (pops.size,), dtype=evecs.dtype)
        valid = mapping >= 0
        rows[valid] = evecs[mapping[valid]]
        result[start:stop] = (rows * pops) @ rows.conj().swapaxes(-1, -2)
    return result


@ut.cache('npz', 'time_evolved_overlaps_and_ps_2norm_v2')
def time_evolved_overlaps_and_ps_2norm(evals, evecs, psi0_eng_basis,
        tau_series, k_series, DA, cond_scr_moment, chunk=32, verbose=False,
        note=None, basis_indices=None, full_dimension=None, symmetric=False):
    """Evolve batches with BLAS and compare projected moments to Scrooge.

    basis_indices maps sector rows back into the full computational basis.
    symmetric must match the representation of cond_scr_moment.
    """
    D, indices, _ = _basis_map(evecs, DA, basis_indices, full_dimension)
    if chunk < 1:
        raise ValueError('chunk must be positive')
    ret = {'overlaps': np.empty(len(tau_series), dtype=np.complex128),
           '2norm': np.empty((len(tau_series), len(k_series)))}
    # Global phase is irrelevant; centering improves large-time phase accuracy.
    centered = evals - evals[0]
    pops = np.abs(psi0_eng_basis)**2
    for start in range(0, len(tau_series), chunk):
        stop = min(start+chunk, len(tau_series))
        if verbose:
            print(f'Projected samples: {start}/{len(tau_series)}', flush=True)
        phases = np.exp(-1j*tau_series[start:stop, None]*centered)
        ret['overlaps'][start:stop] = (phases @ pops) * np.exp(-1j*tau_series[start:stop]*evals[0])
        compact = (phases * psi0_eng_basis) @ evecs.T
        if len(indices) == D and np.array_equal(indices, np.arange(D)):
            states = compact
        else:
            states = np.zeros((stop-start, D), dtype=compact.dtype)
            states[:, indices] = compact
        states = states.reshape(stop-start, -1, DA)
        for j, k in enumerate(k_series):
            moment = projected_moment(states, int(k), symmetric=True)
            target = cond_scr_moment[f'k{k}']
            if not symmetric:
                embedding = symmetric_isometry(DA, int(k))
                target = embedding.T @ target @ embedding
            diff = moment-target
            ret['2norm'][start:stop, j] = np.sqrt(np.sum(np.abs(diff)**2, axis=(-2, -1)))
    return ret


@ut.cache('npz', 'cond_scr_moment_v2')
def cond_scr_moment(evecs, pops, n_scr, k_series, DA, rng, note=None,
                    basis_indices=None, full_dimension=None, symmetric=False,
                    bath_chunk=32):
    """Conditional Scrooge moments, sampled from the probability-adjusted Haar measure.

    k=1 is computed analytically. Higher orders use normalized complex Gaussian
    vectors, which have the Haar distribution without generating full unitaries.
    """
    if n_scr < 1:
        raise ValueError('n_scr must be positive')
    cov = conditional_covariances(evecs, pops, DA, basis_indices,
                                 full_dimension, bath_chunk)
    haar = rng.standard_normal((n_scr, DA)) + 1j*rng.standard_normal((n_scr, DA))
    haar /= np.linalg.norm(haar, axis=-1, keepdims=True)
    ret = {f'k{k}': np.zeros((len(symmetric_basis(DA, int(k))[0]),)*2,
                           dtype=np.complex128) for k in k_series}
    for start in range(0, len(cov), bath_chunk):
        blocks = cov[start:start+bath_chunk]
        values, vectors = np.linalg.eigh(blocks)
        roots = (vectors * np.sqrt(np.clip(values, 0, None))[:, None, :]) @ vectors.conj().swapaxes(-1, -2)
        states = haar @ roots.swapaxes(-1, -2)
        for k in k_series:
            if k == 1:
                ret['k1'] += blocks.sum(axis=0)
            else:
                ret[f'k{k}'] += (DA/n_scr) * projected_moment(states, int(k)).sum(axis=0)
    if not symmetric:
        for k in k_series:
            embedding = symmetric_isometry(DA, int(k))
            ret[f'k{k}'] = embedding @ ret[f'k{k}'] @ embedding.T
    return ret
