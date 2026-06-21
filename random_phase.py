import numpy as np
import itertools as it

def rp_moment(weights, k):
    d = weights.size
    ret = np.zeros((d,)*2*k)
    for idx in np.ndindex((d,)*k):
        perms = set(it.permutations(idx))  # only include unique permtations
        for perm in perms:
            ret[idx + perm] = np.prod(weights[list(idx)])
    return ret.reshape((d**k, d**k))


if __name__ == '__main__':
    d = 2**18
    k = 2
    weights = np.array([1/d]*d)
    
    x = rp_moment(weights, k)
    print(x)
    print(np.einsum('aa->', x))
    print(np.trace(x))
