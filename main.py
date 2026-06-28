import dynamite as dm
import dynamite.states as st
import numpy as np
import matplotlib.pyplot as plt

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

def temp_square_2norm_exact(evals, pops, k_series, tau_series):
    ret = np.zeros((tau_series.size, k_series.size))
    for i, k in enumerate(k_series):
        for n in np.ndindex((2**dm.config.L,) * k):
            print(n)
            for m in np.ndindex((2**dm.config.L,) * k):
                ret[:,i] += np.sinc(np.sum(evals[list(n)] - evals[list(m)]) * tau_series / 2 / np.pi)**2 * np.prod(pops[list(n)] * pops[list(m)])
    return ret

if __name__ == '__main__':
    dm.config.L = 6
    dm.config.initialize(['-mfn_ncv', '80'])
    hx = (np.sqrt(5) + 5) / 8
    hz = (np.sqrt(5) + 1) / 4
    
    k_series = 1 + np.arange(3)

    H = hf.MFIM(hx=hx, hz=hz)
    psi0 = st.State(0)
    
    log_ns = 6
    log_dt = -2
    num_samples = 10**log_ns
    dt = 10**log_dt
    tau_series_sampled = dt * np.arange(num_samples)
    tau_series_exact = np.logspace(1, num_samples, num=10)
    
    print('Computing overlaps')
    overlaps = time_evolved_overlaps(H, psi0, dt, num_samples, note=f'MFIM_L{dm.config.L}_dt{log_dt}_ns{log_ns}')
    print('Computing eigs')
    eigs = la.eig_system(H.to_numpy(sparse=False), note=f'MFIM_L{dm.config.L}_hx{round(hx, 4)}_hz{round(hz, 4)}')
    pops = la.get_pops(eigs['evecs'], psi0.to_numpy())
    
    fig, ax = plt.subplots()
    print('Computing temp_norm_squared_sampled')
    temp_norm_sampled = temp_square_2norm_sampled(overlaps, k_series)
    print('Computing temp_norm_squared_exact')
    # temp_norm_exact = temp_square_2norm_exact(eigs['evals'], pops, k_series, tau_series_exact)

    for i, k in enumerate(k_series):
        rpe_norm = rp.rpe_square_2norm(pops, k)
        diff_sampled = np.sqrt(temp_norm_sampled[:, i] - rpe_norm)
        # diff_exact = np.sqrt(temp_norm_exact[:, i] - rpe_norm)
        ax.plot(tau_series_sampled, diff_sampled, label=rf'$k={k}$, sampled')
        # ax.plot(tau_series_exact, diff_exact, label=rf'$k={k}$, exact')
    ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RPE.}^{(k)}||_2$'
    
    ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
    ax.legend(**ut.LEGEND_OPTIONS)
    fig.savefig('blah.svg', **ut.FIG_SAVE_OPTIONS)

