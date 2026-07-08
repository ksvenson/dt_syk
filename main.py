import dynamite as dm
import dynamite.operators as op
import dynamite.states as st
import numpy as np
import matplotlib.pyplot as plt
import os

import itertools as it
import math

import linalg as la
import hamiltonian_factory as hf
import random_phase as rp

import utilities as ut


@ut.cache('npy', 'time_evolved_overlaps')
def time_evolved_overlaps(H, psi0, dt, num_samples, note=None):
    overlaps = np.full(num_samples, np.nan, dtype='complex')
    psit = psi0.copy()
    overlaps[0] = 1
    for i in np.arange(1, num_samples):
        print(f'{i+1} / {num_samples}')
        psit = dm.computations.evolve(H, psit, dt)
        overlaps[i] = psi0.dot(psit)
    return overlaps

def temp_square_2norm_sampled(overlaps, k_series):
    assert overlaps[0] == 1
    ret = np.full((overlaps.size, k_series.size), np.nan)
    ret[0] = 1
    ret[1:] = np.power.outer(np.abs(overlaps[1:]), 2*k_series)
    helper = np.arange(1, overlaps.size)[:, np.newaxis]
    t1 = np.cumsum(ret[1:], axis=0)
    t2 = np.cumsum(ret[1:] * helper, axis=0)
    ret[1:] = (1 + 2*t1) / (helper+1) - 2*t2 / (helper+1)**2
    return ret

@ut.cache('npy', 'temp_square_2norm_exact')
def temp_square_2norm_exact(evals, pops, k_series, tau_series, chunk=2**10, note=None):
    ret = np.zeros((tau_series.size, k_series.size))
    d = evals.size
    for k_idx, k in enumerate(k_series):
        print(f'k: {k}')
        # Let D_k be the dimension of the symmetric subspace
        multi_sets, mult = np.unique(np.sort(list(np.ndindex((d,)*k)), axis=-1), axis=0, return_counts=True)  # (D_k, k), (D_k,)
        eng_sum = np.sum(evals[multi_sets], axis=-1)                                                          # (D_k,)
        pop_prod = mult * np.prod(pops[multi_sets], axis=-1)                                                  # (D_k,)
        for alpha_idx in np.arange(0, multi_sets.shape[0], chunk):
            alpha_eng_sum = eng_sum[alpha_idx : alpha_idx + chunk]
            alpha_pop_prod = pop_prod[alpha_idx : alpha_idx + chunk]
            for beta_idx in np.arange(alpha_idx, multi_sets.shape[0], chunk):  # sum over upper triangle of (alpha, beta)
                beta_eng_sum = eng_sum[beta_idx : beta_idx + chunk]          # (chunk,)
                beta_pop_prod = pop_prod[beta_idx : beta_idx + chunk]        # (chunk,)

                print(f'({alpha_idx}, {beta_idx}) / {multi_sets.shape[0]}')
                sinc_arg = np.subtract.outer(alpha_eng_sum, beta_eng_sum)
                coeff = np.multiply.outer(alpha_pop_prod, beta_pop_prod)
                if beta_idx == alpha_idx:
                    mask = np.triu_indices(alpha_eng_sum.shape[0], k=1)
                    sinc_arg = sinc_arg[mask]
                    coeff = coeff[mask]
                else:
                    sinc_arg = sinc_arg.flatten()
                    coeff = coeff.flatten()
                sinc_arg = np.multiply.outer(sinc_arg, tau_series / 2)
                ret[:, k_idx] += 2 * np.sum((np.sin(sinc_arg) / sinc_arg)**2 * coeff[:, np.newaxis], axis=0)
                # term = np.subtract.outer(eng_sum, eng_sum[beta_idx : beta_idx + chunk])                           # (D_k, chunk)
                # term = np.sinc(np.multiply.outer(term, tau_series) / 2 / np.pi)**2                                # (D_k, chunk,tau)
                # term *= np.multiply.outer(pop_prod, pop_prod[beta_idx : beta_idx + chunk])[:, :, np.newaxis]      # (D_k, chunk, tau)
                # ret[:, k_idx] += np.sum(term, axis=(0, 1))
    return ret


if __name__ == '__main__':
    # dynamite configuration
    dm.config.initialize(['-mfn_ncv', '80'])
    dm.config.L = 6
    
    # Hamiltonian
    hx = (np.sqrt(5) + 5) / 8
    hz = (np.sqrt(5) + 1) / 4
    H = hf.MFIM(hx=hx, hz=hz)

    # Initial State
    psi0_str = 'mark37'
    psi0 = None
    if psi0_str == '0':
        psi0 = st.State(0)
    elif psi0_str == 'mark37':
        theta = 0.6
        psi0 = dm.computations.evolve(op.index_sum(op.sigmaz()), st.State(0), -theta / 2)
    else:
        print('Unsupported initial state')
        quit()

    base_note = f'MFIM_L{dm.config.L}_hx{hx:.4f}_hz{hz:.4f}_psi0{psi0_str}'
    
    # Sampling parameters
    k_series = 1 + np.arange(3)
    log_ns = 7
    log_dt = -1
    num_samples = 10**log_ns
    dt = 10**log_dt
    exact_num_samples = 20
    tmax = 10**(log_ns + log_dt)
    tau_series_sampled = dt * np.arange(num_samples)
    tau_series_exact = np.logspace(0, log_ns + log_dt, num=exact_num_samples)
    
    # Computations
    print('Computing eigs')
    eigs = la.eig_system(H.to_numpy(sparse=False), note=base_note)
    pops = la.get_pops(eigs['evecs'], psi0.to_numpy())
    print('Computing overlaps')
    overlaps = time_evolved_overlaps(H, psi0, dt, num_samples, note=f'{base_note}_dt{log_dt}_ns{log_ns}')
    
    print('Computing temp_norm_squared_sampled')
    temp_norm_sampled = temp_square_2norm_sampled(overlaps, k_series)
    print('Computing temp_norm_squared_exact')
    temp_norm_exact = temp_square_2norm_exact(eigs['evals'], pops, k_series, tau_series_exact, note=f'{base_note}_tmax{log_ns+log_dt}_ns{exact_num_samples}')
    
    # Plotting
    fig, ax = plt.subplots()
    for i, k in enumerate(k_series):
        rpe_norm = rp.rpe_square_2norm(pops, k)
        diff_sampled = np.sqrt(temp_norm_sampled[:, i] - rpe_norm)
        diff_exact = np.sqrt(temp_norm_exact[:, i])  # we already subtracted the RPE term in this computation
        ax.plot(tau_series_sampled, diff_sampled, label=rf'$k={k}$, sampled', color=f'C{i}')
        ax.plot(tau_series_exact, diff_exact, label=rf'$k={k}$, exact', color=f'black', linestyle='dashed')
    ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RPE.}^{(k)}||_2$'
    
    ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
    ax.legend(**ut.LEGEND_OPTIONS)
    fig.savefig(os.path.join(ut.FIG_DIR, f'{base_note}_dt{log_dt}_ns{log_ns}.svg'), **ut.FIG_SAVE_OPTIONS)

