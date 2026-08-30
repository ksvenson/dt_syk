import string

import numpy as np

import dynamite.computations as comp

import utilities as ut


einsum_idxes = list(string.ascii_lowercase)


def moment_constructor(k):
    """
    Creates a string defining the contraction scheme for the computation of the
    projected ensemble.
    `a` is the tau index.
    `b` is the index for the bath (which we sum over).
    All other indices are for the copies of system A.
    """
    assert 2*k+2 < len(einsum_idxes)
    input = ','.join([f'ab{i}' for i in einsum_idxes[2:2*k+2]])
    output = ''.join(einsum_idxes[2:2*k+2])
    return input + '->a' + output


def norm(arr, p, **kwargs):
    ord = None
    if p == 1:
        ord = 'nuc'
    elif p == 2:
        ord = 'fro'
    else:
        raise NotImplementedError('Only norms 1 and 2 are implemented')
    return np.linalg.norm(arr, ord=ord, **kwargs)


@ut.cache('npz', 'proj_moment')
def proj_moment(H, psi0, tau_series, k_series, DA, note=None):
    psit = [psi0]
    for i, dt in enumerate(np.ediff1d(tau_series, to_begin=tau_series[0])):
        psit.append(comp.evolve(H, psit[-1], dt))
    psit = np.asarray([p.to_numpy(to_all=True) for p in psit[1:]])
    psit = psit.reshape(tau_series.size, -1, DA)
    norm = np.sum(np.abs(psit)**2, axis=-1, keepdims=True)
    out = {}
    for k_idx, k in enumerate(k_series):
        # TODO: post-select on parity
        moment = np.einsum(
            moment_constructor(k),
            *((psit / norm**(1-1/k),)*k + (psit.conj(),)*k),
            optimize=True
        )
        out[str(k)] = moment.reshape(
            tau_series.size,
            DA**k,
            DA**k
        )
    return out

