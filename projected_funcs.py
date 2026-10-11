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
    """Describe the sum used by the original full tensor-basis calculation.

    A kth moment is an average of (|phi><phi|) tensor-powered k times.
    This helper returns the index notation understood by numpy.einsum: the
    first k factors supply ket amplitudes and the last k supply their complex
    conjugates. It does not itself calculate a moment or normalize states.

    Parameters
    ----------
    k : positive integer
        Number of copies of subsystem A.
    ensemble : 'projected' or 'scrooge'
        'projected' keeps the first index (time) and sums the second (bath
        outcome). 'scrooge' sums both (bath outcome and Haar sample).

    Returns
    -------
    str
        A prescription for multiplying amplitudes and summing shared indices.
        Normalization and probability weights must be supplied separately.
        The production calculation uses projected_moment instead; this helper
        is retained to compare with the original formula."""
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
    """Label an orthonormal basis for k identical copies of subsystem A.

    Every vector |phi> tensor-powered k times is unchanged by permutations of
    its copies. It therefore lives in the symmetric subspace, whose dimension
    is S = binomial(DA+k-1, k), rather than the full dimension DA**k.

    A returned tuple t=(a_1,...,a_k), sorted in nondecreasing order, labels a basis
    vector made by equally superposing all distinct permutations of that
    product state. If level a occurs n_a times, there are
    m(t)=k!/product_a(n_a!) distinct permutations. The normalized basis vector
    is their sum divided by sqrt(m(t)).

    Parameters
    ----------
    DA : positive integer
        Hilbert-space dimension of subsystem A, not its number of fermions.
    k : positive integer
        Number of copies in the moment.

    Returns
    -------
    tuples : integer array, shape (S, k)
        Basis labels in a fixed order. Repeated levels are allowed, e.g.
        for DA=2,k=2 the labels are (0,0), (0,1), (1,1).
    factors : real array, shape (S,)
        sqrt(m(t)) for each label. The coefficient of |phi>**tensor k along
        that basis vector is factors[t]*product_j(phi[a_j]). In the example
        above the coefficients are phi[0]**2, sqrt(2)*phi[0]*phi[1], phi[1]**2.

    Notes
    -----
    The returned arrays are reused by later calls; treat them as read-only.
    This basis reduction is exact and makes no assumption about thermalization."""
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
    """Express symmetric-subspace basis vectors in the full product basis.

    Parameters
    ----------
    DA : positive integer
        Dimension of one copy of subsystem A.
    k : positive integer
        Number of copies.

    Returns
    -------
    U : real array, shape (DA**k, S), S=binomial(DA+k-1,k)
        Each column is one normalized symmetric basis vector, ordered as in
        symmetric_basis. Rows are the ordinary product basis in the order
        used by NumPy's reshape: the last copy's index changes fastest.

    Verification
    ------------
    U.conj().T @ U = identity(S). If M is a moment in symmetric coordinates,
    its full tensor-basis matrix is U @ M @ U.conj().T. If two operators are
    supported on this symmetric subspace, their Hilbert-Schmidt distance is
    identical before and after this conversion. Here U is real, so U.T is
    also its conjugate transpose.

    Notes
    -----
    This is mainly for verification and compatibility with full-basis targets.
    Constructing U still requires DA**k rows. Its returned array is reused;
    treat it as read-only."""
    tuples, factors = symmetric_basis(DA, k)
    embedding = np.zeros((DA**k, len(tuples)))
    for j, t in enumerate(tuples):
        for p in set(it.permutations(t)):
            embedding[np.ravel_multi_index(p, (DA,)*k), j] = 1/factors[j]
    return embedding


def weighted_features(states, k):
    """Represent unnormalized conditional k-copy states with their weights.

    For an input ket |v>, let p=<v|v>. For p>0, this function returns the
    symmetric-basis amplitudes of |F>=sqrt(p)*(|v>/sqrt(p))**tensor k.
    Consequently |F><F|=p*(|v><v|/p)**tensor k, exactly the Born-weighted
    contribution needed in a projected moment. A zero ket returns a zero
    vector and makes no contribution; no division by zero is performed.

    Parameters
    ----------
    states : floating-point or complex array, shape (..., DA)
        The last index is the computational-basis amplitude within A.
        Preceding indices may label times, bath outcomes, or Haar samples.
        These are unnormalized kets; the norm supplies the probability weight.
        For Scrooge integration the same mathematical operation is used, but
        p is an integration weight, not by itself a normalized probability.
    k : positive integer
        Moment order.

    Returns
    -------
    features : array, shape (..., S), S=binomial(DA+k-1,k)
        The amplitudes of F, in the order given by symmetric_basis.
        Its squared norm equals p for every input ket and every k.
        No sum over outcomes or samples is performed here."""
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
    """Compute Born-weighted moments from conditional subsystem kets.

    Suppose |Psi>=sum_z |z>_B tensor |v_z>_A. Measuring B in the computational
    basis gives outcome z with probability p_z=<v_z|v_z> and leaves A in
    |phi_z>=|v_z>/sqrt(p_z). This function calculates

        M_k = sum_z p_z (|phi_z><phi_z|)**tensor k.

    A zero-probability outcome contributes zero. The input is not normalized
    by this function: Tr(M_k)=sum_z p_z, which equals one for a normalized
    global state. The same sum can be used as a building block for Scrooge
    integration; its extra integration factor is applied by cond_scr_moment.

    Parameters
    ----------
    states : floating-point or complex array, shape (..., outcomes, DA)
        Each states[...,z,:] is the unnormalized ket v_z. The second-to-last
        index is summed. Earlier indices, such as time, remain in the result.
    k : positive integer
        Moment order.
    symmetric : bool, default True
        Return the operator in the orthonormal symmetric basis if True,
        or the ordinary k-copy product basis if False. Both describe the
        same physical operator and give the same Hilbert-Schmidt distances.

    Returns
    -------
    moment : complex or real array
        Shape (..., S, S) if symmetric=True, or (..., DA**k, DA**k) otherwise.
        Matrix entry (a,b) is sum_z F_z[a]*conj(F_z[b]). For k=1 this is
        the reduced density matrix Tr_B(|Psi><Psi|).

    Notes
    -----
    This calculation is exact up to floating-point arithmetic. It does not
    sample bath outcomes: it sums all supplied outcomes."""
    features = weighted_features(states, k)
    moment = features.swapaxes(-1, -2) @ features.conj()
    if not symmetric:
        embedding = symmetric_isometry(states.shape[-1], k)
        moment = embedding @ moment @ embedding.T
    return moment


def _basis_map(evecs, DA, basis_indices, full_dimension):
    """Relate a parity-sector basis to the full computational basis.

    A parity-sector vector has fewer entries than a full-system ket. Its row
    number is not in general a bath/subsystem basis label. This helper checks
    an explicit correspondence before any subsystem projection is made.

    Parameters
    ----------
    evecs : array, shape (number_of_sector_states, number_of_energy_states)
        Energy eigenvectors as columns in the retained computational basis.
    DA : positive integer
        Dimension of A; the full dimension must be divisible by DA.
    basis_indices : integer array or None
        basis_indices[j] is the full computational-basis label of sector row j.
        None means the rows already use the ordinary full basis.
    full_dimension : integer or None
        Dimension D of A tensor B. None uses the number of eigenvector rows.
        For a reduced parity sector, supply both this value and basis_indices.

    Returns
    -------
    D : integer
        Full Hilbert-space dimension.
    indices : integer array, shape (number_of_sector_states,)
        The checked sector-row to full-basis correspondence.
    inverse : integer array, shape (D,)
        The sector row corresponding to a full-basis label, or -1 if that
        label is outside the retained sector. Missing amplitudes are zero.

    Notes
    -----
    With the convention used throughout this repository, a full-basis label
    is z*DA+a, where z labels B and a labels A."""
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
    """Form the unnormalized conditional density matrices sigma_{A|z}.

    Define sigma=sum_alpha p_alpha |E_alpha><E_alpha|. For each computational
    measurement outcome z of B, calculate

        sigma_{A|z}[a,b] = sum_alpha p_alpha
                           <z,a|E_alpha><E_alpha|z,b>.

    This is <z|sigma|z> on A. It is not divided by its trace. Its trace q_z
    is the probability of outcome z in sigma, and sigma_{A|z}/q_z is the
    normalized conditional density matrix when q_z>0. Summing these matrices
    over z gives Tr_B(sigma). Despite the function name, these are quantum
    density-matrix blocks, not classical statistical covariance matrices.

    Parameters
    ----------
    evecs : complex array, shape (sector_dimension, number_of_energy_states)
        Columns are energy eigenkets; rows use the retained computational basis.
    pops : real array, shape (number_of_energy_states,)
        Nonnegative energy populations p_alpha, summing to one for a state.
    DA : positive integer
        Dimension of A. The ordering is |z>_B tensor |a>_A, label z*DA+a.
    basis_indices, full_dimension : array and integer, or None
        Correspondence to the full basis, as described in _basis_map.
        Supply these when the eigenvectors are restricted to even parity.
    bath_chunk : positive integer, default 32
        Number of bath outcomes calculated together. This controls temporary
        storage; it does not alter the physics or approximate the bath sum.

    Returns
    -------
    result : complex array, shape (DB, DA, DA), DB=full_dimension/DA
        result[z] is sigma_{A|z}. Forbidden sector amplitudes are set to zero.
        The calculation forms only these needed blocks, not the full sigma.

    Notes
    -----
    sigma is defined as diagonal in the supplied energy basis. If energy
    degeneracies retain coherences, it need not equal the actual infinite-time
    average of |psi(t)><psi(t)|; those coherences are not included here."""
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
    """Compare each evolved state's projected ensemble with a fixed target.

    For each supplied time t, form |psi(t)>=sum_alpha c_alpha exp(-i E_alpha t)
    |E_alpha>, then measure B in its computational basis. All bath outcomes
    are summed exactly in the resulting projected moment M_k(t).

    Parameters
    ----------
    evals : real array, shape (number_of_energy_states,)
        Energies E_alpha, in units with hbar=1.
    evecs : complex array, shape (sector_dimension, number_of_energy_states)
        Corresponding eigenkets as columns, in the retained basis.
    psi0_eng_basis : complex array, shape (number_of_energy_states,)
        c_alpha=<E_alpha|psi(0)>; squared magnitudes must sum to one.
    tau_series : real array, shape (number_of_times,)
        Actual evolution times. They need not be uniformly spaced. In the
        production script these are randomly chosen projected sample times,
        not the endpoint times at which running averages are reported.
    k_series : array of positive integers
        Moment orders to compare, in the desired output-column order.
    DA : positive integer
        Dimension of A, using full-basis ordering z*DA+a.
    cond_scr_moment : dictionary
        Target operators under keys 'k1', 'k2', etc., normally the output of
        the function of the same name. No target is estimated in this function.
    chunk : positive integer, default 32
        Number of times calculated together; limits temporary storage only.
    verbose : bool
        Print progress if True.
    note : str or None
        Name used to save/reuse the calculation. Saved results are identified
        by this name, not by checking every input array. Use a distinct name
        whenever any physical parameter, sampling rule, or target changes.
    basis_indices, full_dimension : array and integer, or None
        Sector-to-full basis correspondence; see _basis_map. Each evolved
        sector ket is expanded with zero amplitudes outside the sector before
        reshaping into bath outcomes and subsystem amplitudes.
    symmetric : bool, default False
        Describes the supplied target's representation: True for symmetric
        coordinates, False for the full tensor basis. Must match the target.
        The actual distance calculation uses symmetric coordinates in both cases.

    Returns
    -------
    ret : dictionary
        'overlaps': complex array of shape (number_of_times,), containing
        <psi(0)|psi(t)>, including its overall phase.
        '2norm': real array of shape (number_of_times, len(k_series)), containing
        sqrt(Tr[(M_k(t)-target_k)^dagger (M_k(t)-target_k)]).
        This is the Hilbert-Schmidt/Frobenius norm, not the trace norm or the
        largest-singular-value norm sometimes also called a matrix 2-norm.

    Notes
    -----
    Energies are shifted by evals[0] while forming projected states. This only
    removes an irrelevant overall phase; that phase is restored in overlaps.
    No integration over time occurs here: the caller subsequently averages
    the returned distances. Evolution and the bath-outcome sum are exact up
    to floating-point error; a sampled Scrooge target carries its own error."""
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
    """Calculate the bath-probability-weighted conditional Scrooge moments.

    Let sigma=sum_alpha p_alpha |E_alpha><E_alpha| and sigma_z=<z|sigma|z>.
    The target is C_k=sum_z q_z S_k(sigma_z/q_z), where q_z=Tr(sigma_z)
    and S_k denotes the kth moment of the Scrooge ensemble. Terms with q_z=0
    are zero. This is a mixture of conditional Scrooge ensembles, not a
    single Scrooge ensemble built from sum_z sigma_z.

    For a Haar-uniform normalized ket |h> in the full DA-dimensional space,
    set |v_z>=sqrt(sigma_z)|h>. Then the same target can be written

        C_k = DA * sum_z integral d h <v_z|v_z>
                         (|v_z><v_z|/<v_z|v_z>)**tensor k.

    Thus the norm supplies an extra probability weight. Simply averaging
    normalized sqrt(sigma_z)|h> states without this weight would give a
    different ensemble. Factors q_z are already included through sigma_z;
    the code does not multiply by them again.

    Parameters
    ----------
    evecs, pops, DA, basis_indices, full_dimension
        Eigenkets, energy populations, subsystem dimension, and optional
        sector correspondence as described in conditional_covariances.
    n_scr : positive integer
        Number of independent Haar input kets. Each is obtained by normalizing
        a vector with independent complex Gaussian components. The same list
        is reused for every bath outcome; sampling errors between bath
        outcomes are therefore correlated, while the expectation is correct.
    k_series : array of positive integers
        Requested moment orders.
    rng : numpy.random.Generator
        Random number source for the Haar kets. Reuse its seed to reproduce
        the estimate when the calculation is not loaded from saved results.
    note : str or None
        Name identifying saved results; change it when physical inputs,
        representation, sample count, or random seed change.
    symmetric : bool, default False
        True returns operators in the orthonormal symmetric k-copy basis;
        False returns full product-basis operators.
    bath_chunk : positive integer, default 32
        Number of bath outcomes processed together; controls temporary storage.

    Returns
    -------
    ret : dictionary
        ret['k{k}'] is C_k, with shape (S,S) if symmetric=True, where
        S=binomial(DA+k-1,k), or (DA**k,DA**k) otherwise.
        k=1 is evaluated exactly as sum_z sigma_z=Tr_B(sigma).
        For k>1, the integral is estimated by DA/n_scr times a sum over
        the n_scr Haar inputs and all bath outcomes.

    Notes
    -----
    Small negative eigenvalues of sigma_z from floating-point roundoff are
    set to zero before taking its positive square root. Rank-deficient
    blocks and zero-weight outcomes are allowed. Higher-order estimates are
    not divided by their measured trace, so their traces can fluctuate around
    one. Sampling uncertainty is not calculated or returned by this function."""
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
