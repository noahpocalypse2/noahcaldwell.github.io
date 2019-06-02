# M472_Proj3_iterativemethods.py
# M472 Dr.Karakashian
# written by Noah Caldwell
# 5/2/19
# Performs method of steepest descent and Conjugate Gradient

import numpy as np
import math
import timeit
global d
d = 2
global s
s = -1

def Anorm(x, n):
    return np.sqrt(float( sum([d*x[i]*x[i] for i in range(0,n)]) + 2*sum([s*x[i]*x[i+1] for i in range(0,n-1)])))

def getA(n):
    A = np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                A[i,j] = 2
            if i + 1 == j or i == j+1:
                A[i,j] = -1
    return A

def steepest_descent(n, maxIT):
    h = 1 / (n+1)
    b = np.matrix([2 * h * h for i in range(1,n+1)]).T
    full_solution = np.matrix([i * h * (1 - i * h) for i in range(1,n+1)]).T
    x = np.zeros([n, 1])
    end_criteria = 0.00001 * Anorm(x - full_solution, n)
    prevx = x
    rk = b
    alpha = float( (rk.T @ rk) / (rk.T @ A @ rk) )
    k = 0
    while k < maxIT:
        k += 1
        
        x = x + (alpha * rk)
        rk = rk - alpha * (A@rk)
        alpha = float( (rk.T @ rk) / (rk.T @ A @ rk) )
        prevx = x
        err = Anorm(x - full_solution, n)
        if err < end_criteria:
            print("Err < end criteria!")
            break
            
    print("DONE with steepest descent method: {} iterations, error = {}".format(k, err))
    return k, x, full_solution

def conjugate_gradient(n, maxIT, A):
    h = 1 / (n+1)
    b = np.matrix([2 * h * h for i in range(1,n+1)]).T
    full_solution = np.matrix([i * h * (1 - i * h) for i in range(1,n+1)]).T
    x = np.zeros([n,1])
    end_criteria = 0.00001 * Anorm(x - full_solution, n)
    r = np.matrix([1])
    r = []
    r.append(b)
    k = 0
    while k < maxIT:
        k += 1
        if k == 1:
            p = [0]
            p.append(r[0])
        else:
            beta = float( (r[k-1].T @ r[k-1]) / (r[k-2].T @ r[k-2]) )
            p.append(r[k-1] + beta * p[k-1])
        q = A @ p[k]
        alpha = float( (r[k-1].T @ r[k-1]) / (p[k].T @ q) )
        x = x + alpha * p[k]
        r.append(r[k-1] - alpha * q)
        err = Anorm(x - full_solution, n)
        if err < end_criteria:
            print("Err < end criteria!")
            break
    print("DONE with conjugate gradient: {} iterations, error = {}".format(k, err))
    return k, x, full_solution
    

n = [25, 50, 100, 200]
maxIT = 10000000

print("Steepest descent first:")
for i in range(0,4):
    print("n = {}".format(n[i]))
    A = getA(n[i])
    eigvalues = np.linalg.eigvals(A)
    condition_number = max(eigvalues) / min(eigvalues)
    print("Condition number = {}".format(condition_number))
    cn_sqrt = np.sqrt(condition_number)
    start = timeit.default_timer()
    
    k, x, full_solution = steepest_descent(n[i], maxIT)
    ma = Anorm(x - full_solution, n[i]) / Anorm(-1 * full_solution, n[i])
    mal = math.log(ma, k)
    kappa = mal / (mal + 1)
    print("Kappa = {}".format(kappa))
    
    k, x, full_solution = conjugate_gradient(n[i], maxIT, A)
    ma = Anorm(x - full_solution, n[i]) / Anorm(-1 * full_solution, n[i])
    mal = math.log( (1 / np.sqrt(2)) * ma, k)
    kappa = mal / (mal + 1)
    kappa = kappa*kappa
    print("Kappa = {}".format(kappa))

    stop = timeit.default_timer()
    print("Total time elapsed = {} seconds".format(stop-start))

