import dynamite as dt
import dynamite.states as st
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import hamiltonian_factory as hf
import random_phase as rp

import utilities as ut



def evolve(H, psi0, t):
    """
    Wrapper for dynamite's evolve function so we can cache the result.
    """
    return dt.computations.evolve(H, psi0, t)

def time_evolved_overlaps(H, time):
    overlaps = np.zeros(time.shape, dtype='complex')
    for i, t in enumerate(time):
        psit = evolve(H, psi0, t)
        overlaps[i] = psi0.dot(psit)
    return overlaps

def temp_square_2norm(H, psi0, k, max_steps, step_width):
    # time needs to be evenly spaced, starting at zero.
    time = step_width * np.arange(max_steps)
    expo_overlaps = np.abs(time_evolved_overlaps(H, time[1:]))**(2*k)
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

def rpe_square_2norm(H, psi0, k):
    eigs = eig_system(H.to_numpy(sparse=False))
    evecs = eigs['evecs']
    pops = get_pops(evecs, psi0)


if __name__ == '__main__':
    dt.config.L = 6
    dt.config.initialize(['-mfn_ncv', '80'])

    H = hf.MFIM()
    psi0 = st.State(0)
    pops = np.load('./pops.npy')

    max_steps = 5
    step_width = 100

    k_list = 1 + np.arange(3)

    fig, ax = plt.subplots()
    for k in k_list:
        time = step_width * np.arange(max_steps)
        temp_norm = temp_square_2norm(H, psi0, k, max_steps, step_width)
        rpe_norm = rp.rpe_square_2norm(pops, k)
        diff = np.sqrt(temp_norm - rpe_norm)

        ax.plot(time, diff, label=rf'$k={k}$')
    # ylabel = r'$\text{Tr}((\rho_\text{Temp.}^{(k)})^2)$'
    ylabel = r'$||\rho_\text{Temp.}^{(k)} - \rho_\text{RPE.}^{(k)}||_2$'
    ax.set(xlabel='Time', ylabel=ylabel, xscale='log', yscale='log')
    ax.legend(**ut.LEGEND_OPTIONS)
    fig.savefig('blah.svg', **ut.FIG_SAVE_OPTIONS)

