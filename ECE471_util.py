# ECE471_util.py
# ECE471 Dr.Qi
# written by Noah Caldwell
# 5/6/19
# Used in final project.

 import numpy as np

def euc2(x, y):
    # calculate squared Euclidean distance

    # check dimension
    assert x.shape == y.shape  

    diff = x - y

    return np.dot(diff, diff)


def mah2(x, y, Sigma):
    # calculate squared Mahalanobis distance

    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    
    diff = x - y
    
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)


def mink(u, v, p=2):
    # calculate Minkowski norm
    if p < 1:
        raise ValueError("p must be at least 1")
    dist = np.linalg.norm(u - v, ord=p)
    return dist
