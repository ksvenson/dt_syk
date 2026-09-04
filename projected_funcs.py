"""
IMPORTANT: this module has an implict convention for factoring the subsystem (A), and
the bath (B). The convention is: an axis of dimension `2**L` is reshaped to `(-1, DA)`.
This must be kept consistent. - Kai Svenson, Aug. 31, 2026
"""
import string

import numpy as np
import scipy as sp

import dynamite.computations as comp

import utilities as ut


einsum_idxes = list(string.ascii_lowercase)


def moment_constructor(k, ensemble):
    """
    Creates a string defining the contraction scheme for the computation of the
    projected  and scrooge ensembles.
    `a` is the tau index (projected ensemble) or index for system B (scrooge ensemble).
    `b` is the index for system B (projected ensemble) or the index of the haar state
        sample (scrooge ensemble). In both cases, this index is summed over, and this
        index represents the probability space for the respective ensemble. 
    All other indices are for the copies of system A.

    This order is an implicit reference to the factoring convention described at the top
    of this module.

    The only difference is that for the projected ensemble, we *don't* sum over the tau
    index (`a`), while for the scrooge ensemble, we sum over the index for system B
    (`a`).
    """
    assert 2*k+2 < len(einsum_idxes)
    input = ','.join([f'ab{i}' for i in einsum_idxes[2:2*k+2]])
    output = ''.join(einsum_idxes[2:2*k+2])
    if ensemble == 'projected':
        return input + '->a' + output
    elif ensemble == 'scrooge':
        return input + '->' + output
    else:
        raise ValueError(f'Unrecognized ensemble: "{ensemble}"!')


@ut.cache('npz', 'time_evolved_overlaps_and_rhoA')
def time_evolved_overlaps_and_rhoA(evals, evecs, psi0, tau_series, k_series, DA, chunk=2**12, note=None):
    """
    This functions computes three objects:
    1. Time evolved overlaps: <psi_0|psi_t>.
       An array with the same size as `tau_series`.
       Stored in `ret['overlaps']`.

    2. Energy eigenstate populations: |<psi_0|E_n>|**2.
       An array with size 2**L.
       Stored in `ret['pops']`.

    3. kth moments of the projected ensemble: \rho_A^{(k)}.
       An array with shape (`tau_series.size`, DA**k, DA**k).
       Stored in `ret[f'rhoA_k{k}]`.

    These objects are bunched in the same computation since all of them except `pops`
    require a chunked loop over time. In particular, \rho_A^{(k)} requires psit, so
    it's more memory efficient to compute \rho_A^{(k)} at the same time the overlaps are
    computed.
    """
    ret = {'overlaps': np.empty(tau_series.shape, dtype=psi0.dtype)}
    for k in k_series:
        ret[f'rhoA_k{k}'] = np.empty(tau_series.shape + (DA**k, DA**k), dtype=psi0.dtype)

    psi0_eng_basis = np.einsum('ab,b->a', evecs.conj().T, psi0)
    ret['pops'] = np.abs(psi0_eng_basis)**2

    for t_idx in range(0, tau_series.size, chunk):
        # Overlap computation
        psit_eng_basis = np.exp(-1j * np.einsum(
            'a,b->ab',
            tau_series[t_idx:t_idx+chunk],
            evals
        )) * psi0_eng_basis[np.newaxis, :]
        ret['overlaps'][t_idx:t_idx+chunk] = np.einsum('a,ba->b', psi0_eng_basis.conj(), psit_eng_basis)

        # rhoA_k computation
        psit = np.einsum('ab,cb->ca', evecs, psit_eng_basis)
        psit = psit.reshape(tau_series.size, -1, DA)  # implicit reference to factoring convention.
        norm2 = np.sum(np.abs(psit)**2, axis=-1, keepdims=True)
        for k in k_series:
            # in the denominator, we have k powers of norm for each of the k-copies of
            # the state, minus 1 power for the probability of getting the projected
            # state. Hence the power of (1-1/k) below, which after the einsum is
            # k*(1-1/k) = k-1.
            ret[f'rhoA_k{k}'][t_idx:t_idx+chunk] = np.einsum(
                moment_constructor(k, 'projected'),
                *((psit / norm2**(1-1/k),)*k + (psit.conj(),)*k),
                optimize=True
            )
            ret[f'rhoA_k{k}'][t_idx:t_idx+chunk] = ret[f'rhoA_k{k}'].reshape(tau_series.size, DA**k, DA**k)
    return ret


@ut.cache('npz', 'cond_scr_moment')
def cond_scr_moment(evecs, pops, n_scr, k_series, DA, rng, note=None):
    """
    "Conditioned Scrooge Moment". See statement of theorem 2 in 2601.00266 (Nature is
    Stingy).
    """
    ret = {}
    D = pops.size
    DB = D // DA
    # sig_A_cond_z in the computational basis (will be left unnormalized)
    sig_A_cond_z = np.einsum('ab,bc', evecs, pops[:, np.newaxis] * evecs.conj().T)
    # expose indices for systems A and B
    sig_A_cond_z = sig_A_cond_z.reshape(DB, DA, DB, DA)
    # project onto z
    sig_A_cond_z = np.einsum('abac->abc', sig_A_cond_z)

    sig_evals, sig_evecs = np.linalg.eigh(sig_A_cond_z)
    sqrt_sig_A_cond_z = np.einsum('zab,zbc', sig_evecs, np.sqrt(sig_evals)[:, np.newaxis] * sig_evecs.conj().T)

    # Generate ranom unitaties and have them act on [1, 0].
    # Can't use the second column since the columns are orthogonal and thus not
    # independent.
    haar_states = sp.stats.unitary_group.rvs(dim=DA, size=n_scr)[:, :, 0]
    scr_states = np.einsum('zab,ub->zua', sqrt_sig_A_cond_z, haar_states)
    norm2 = np.sum(np.abs(scr_states)**2, axis=-1, keepdims=True)

    for k in k_series:
        # in the denominator, we have k powers of norm for each of the k-copies of the
        # state, minus 1 power for the scrooge probability measure. See
        # https://arxiv.org/pdf/2601.00266 (Nature is Stingy, Eq. 11) which has an
        # explicit construction of the scrooge ensemble in terms of the haar measure.
        # Hence we have a power of (1-1/k) below, which after the einsum is k*(1-1/k) =
        # k-1. Interestingly, this is similar to the construction of the projected
        # ensemble.
        # 
        # All factors of <z|\sigma_B|z> cancel out.
        # 
        # The factor of DA comes from Eq. 11. The factor of 1/n_scr comes from averaging over
        # n_scr haar states.
        ret[f'k{k}'] = (DA / n_scr) * np.einsum(
            moment_constructor(k, 'scrooge'),
            *((scr_states / norm2**(1-1/k),)*k + (scr_states.conj(),)*k),
            optimize=True
        )
        ret[f'k{k}'] = ret[f'k{k}'].reshape(DA**k, DA**k)
    return ret


