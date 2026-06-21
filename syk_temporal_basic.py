import itertools
from itertools import combinations
import random
import numpy as np
import scipy
from scipy.special import binom
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from math import factorial, comb
from dynamite.extras import majorana
from dynamite.operators import op_sum, op_product, identity, sigmaz, sigmax
from dynamite.states import State
from dynamite.subspaces import Subspace
from dynamite.computations import evolve
from dynamite import config
import sys 
#from mpi4py import MPI

def main():
    N = 12
    q = 4
    D = 112
    LA = 2
    config.L = N//2
    L = N//2
    t_steps = [i for i in range(10)]
    seed = 42
    rng = np.random.default_rng(seed)

    iState = State(0)
    H = Hamil(N,q,seed)
    iSnp = iState.to_numpy() #to_all not in docs for some reason. w/o an argument, just puts the numpy state on core 0

    print('🚀🚀🚀DONE🚀🚀🚀')
########################################################################################################################################
def Hamil(N, q, random_seed):
    #Fix seed to generate random couplings Jijkl
    rng = np.random.default_rng(random_seed) #use local random state instead of global (global = np.random.seed)
    
    comb = combinations(np.arange(N), q)
    hyperedges = tuple([i for i in comb])
    
    # Use variance with convention J=1
    couplings = (1j)**(q/2)*np.sqrt( factorial(q-1) / (N**(q-1) * 2**q) )*rng.standard_normal(len(hyperedges))
    
    # Create a dictionary to map a hyperedge to the random coupling
    factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i) for i in range(N)]

    return op_sum((op_product(majs[i] for i in idxs)*factor[idxs] for idxs in hyperedges), nshow=len(hyperedges))
############################################################################################################################################


main()
