import dynamite as dm
import dynamite.states as st
import numpy as np
import matplotlib.pyplot as plt

import hamiltonian_factory as hf
import random_phase as rp

import utilities as ut



def evolve(H, psi0, t):
    """
    Wrapper for dynamite's evolve function so we can cache the result.
    """
    return dm.computations.evolve(H, psi0, t)

def time_evolved_overlaps(H, psi0, dt, num_samples):
    overlaps = np.full(num_samples, np.nan, dtype='complex')
    psit = psi0.copy()
    overlaps[0] = 1
    for i in np.arange(1, num_samples):
        print(f'{i+1} / {num_samples}')
        psit = evolve(H, psit, dt)
        overlaps[i] = psi0.dot(psit)
    return overlaps

def temp_square_2norm(overlaps, k_series, num_samples):
    ret = np.full((num_samples, k_series.size), np.nan)
    ret[0] = 1
    ret[1:] = np.power.outer(np.abs(overlaps), 2*k_series)
    helper = np.arange(1, num_samples)[:, np.newaxis]
    t1 = np.cumsum(ret[1:], axis=0)
    t2 = np.cumsum(ret[1:] * helper, axis=0)
    ret[1:] = (1 + 2*t1) / (helper+1) - 2*t2 / (helper+1)**2
    return ret


if __name__ == '__main__':
    dm.config.L = 6
    dm.config.initialize(['-mfn_ncv', '80'])
    
    k_series = 1 + np.arange(3)

    H = hf.MFIM()
    psi0 = st.State(0)

    num_samples = 10000
    dt = 0.1
    tau_series = dt * np.arange(num_samples)

    overlaps = time_evolved_overlaps(H, psi0, dt, num_samples)
    
    pops = np.load('./pops.npy')

    fig, ax = plt.subplots()
    temp_norm = temp_square_2norm(overlaps, k_series, num_samples)

    for i, k in enumerate(k_series):
        rpe_norm = rp.rpe_square_2norm(pops, k)
        diff = np.sqrt(temp_norm[:, i] - rpe_norm)
        ax.plot(tau_series, diff, label=rf'$k={k}$')
    # ax.plot(time, 10**7/time, label=r'$\tau^{-1}$')
    # ax.plot(time, (10**7/time)**(1/2), label=r'$\tau^{-1/2}$')
    # ylabel = r'$\text{Tr}((\rho_\text{Temp.}^{(k)})^2)$'
    ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RPE.}^{(k)}||_2$'
    
    ax.set(xlabel=r'$\tau$', ylabel=ylabel, xscale='log', yscale='log')
    # ax.set(xlabel=r'$\tau$', ylabel=ylabel)
    ax.legend(**ut.LEGEND_OPTIONS)
    fig.savefig('blah.svg', **ut.FIG_SAVE_OPTIONS)

