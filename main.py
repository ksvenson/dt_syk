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
        psit = evolve(H, psit, dt)
        overlaps[i] = psi0.dot(psit)
    return overlaps

def temp_square_2norm(H, psi0, k_series, dt, num_samples):
    overlaps = time_evolved_overlaps(H, psi0, dt, num_samples)[1:]
    ret = np.full((num_samples, k_series.size), np.nan)
    ret[0] = 1
    ret[1:] = np.power.outer(np.abs(overlaps), 2*k_series)
    ret[1:] = np.arange(1, num_samples) * ret
    ret[1:] = np.cumsum(ret[1:], axis=0)
    ret[1:] = (1 - 2*ret[1:]) / np.arange(2, num_samples+1)
    return ret

def temp_square_2norm_old(H, psi0, k_series, tau_series, num_samples):
    ret = np.full((tau_series.size, k_series.size), np.nan)
    for i, tau in enumerate(tau_series):
        dt = tau / (num_samples - 1)
        exp_overlaps = np.power.outer(np.abs(time_evolved_overlaps(H, psi0, dt, num_samples)[1:]), 2*k_series)
        triangle_sum = np.sum(exp_overlaps * np.arange(num_samples - 1, 0, -1)[:, np.newaxis], axis=0)
        ret[i] = (num_samples + 2*triangle_sum) / num_samples**2
    return ret

def temp_square_2norm_old_old(overlaps, k):
    # time needs to be evenly spaced, starting at zero.
    # time = step_width * np.arange(max_steps)
    # expo_overlaps = np.abs(time_evolved_overlaps(H, time[1:]))**(2*k)
    expo_overlaps = np.abs(overlaps)**(2*k)
    ret = np.full(max_steps, np.nan)
    for i in np.arange(max_steps):
        triangle_sum = np.sum(expo_overlaps[:i] * np.arange(i, 0, -1))
        ret[i] = ((i+1) + 2*triangle_sum) / (i+1)**2
    
    # expo_overlaps = np.concatenate(([1], expo_overlaps))
    # ret2 = np.full(max_steps, np.nan)
    # for i in np.arange(max_steps):
    #     mask = np.abs(np.subtract.outer(np.arange(i+1), np.arange(i+1)))
    #     ret2[i] = np.sum(expo_overlaps[mask]) / (i+1)**2

    # ret3 = np.full(max_steps, np.nan)
    # for i in np.arange(max_steps):
    #     ret3[i] = 0
    #     for idx in np.ndindex((i+1, i+1)):
    #         ret3[i] += expo_overlaps[np.abs(idx[0]-idx[1])]
    #     ret3[i] /= (i+1)**2

    return ret


if __name__ == '__main__':
    dm.config.L = 4
    dm.config.initialize(['-mfn_ncv', '80'])
    
    k_series = 1 + np.arange(3)

    H = hf.MFIM()
    psi0 = st.State(0)

    # max_steps = 10**5
    # # step_width = 10**7 * 2**dm.config.L / max_steps
    # step_width = 10**3
    # time = step_width * np.arange(max_steps)

    tau_series = 10**(1 + np.arange(8))
    num_samples = 10000

    # overlaps = time_evolved_overlaps(H, psi0, time[1:])
    
    pops = np.load('./pops.npy')

    fig, ax = plt.subplots()
    temp_norm = temp_square_2norm(H, psi0, k_series, tau_series, num_samples)

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

