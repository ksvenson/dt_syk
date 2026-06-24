import dynamite as dt
import dynamite.states as st
import numpy as np
import matplotlib.pyplot as plt

import hamiltonian_factory as hf

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
    ret = np.zeros((max_steps,))
    for i in np.arange(max_steps):
        triangle_sum = np.sum(expo_overlaps[:i] * np.arange(i, 0, -1))
        ret[i] = ((i+1) + 2*triangle_sum) / (i+1)**2
    return ret


if __name__ == '__main__':
    dt.config.L = 6
    dt.config.initialize(['-mfn_ncv', '80'])

    H = hf.MFIM()
    psi0 = st.State(0)

    max_steps = 100
    step_width = 10**7 / max_steps

    k_list = 1 + np.arange(3)

    fig, ax = plt.subplots()
    for k in k_list:
        norm = temp_square_2norm(H, psi0, k, max_steps, step_width)
        time = step_width * np.arange(max_steps)

        ax.plot(time, norm - norm[-1], label=rf'$k={k}$')
    ax.set(xlabel='Time', ylabel=r'$\text{Tr}((\rho_\text{Temp.}^{(k)})^2)$', xscale='log', yscale='log')
    ax.legend(**ut.LEGEND_OPTIONS)
    fig.savefig('blah.svg', **ut.FIG_SAVE_OPTIONS)

