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
    d = 3
    k = 5
    weights = np.array([0.25, 0.25, .5])
    
    x = rp_moment(weights, k)
    print(x)
    print('Trace:')
    print(np.einsum('aa->', x))
    print('Trace of square:')
    print(np.einsum('ab,ba->', x, x))
    
    trace = 0
    for idx in np.ndindex((d,)*k):
        # print(trace)
        trace += len(set(it.permutations(idx))) * np.prod(weights[list(idx)])**2
    print(trace)
