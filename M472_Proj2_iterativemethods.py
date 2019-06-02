# M472_Proj2_iterativemethods.py
# M472 Dr.Karakashian
# written by Noah Caldwell
# 4/11/19
# Performs Jacobi, Gauss-Seidel, or SoR

import numpy as np
import math

def getA(n):
    A = np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                A[i,j] = 2
            if i + 1 == j or i == j+1:
                A[i,j] = -1
    return A
def getD(n):
    D = np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                D[i,j] = 2
    return D
def getL(n):
    L = np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            if i == j + 1:
                L[i,j] = 1
    return L
def getU(n):
    U = np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            if i + 1 == j:
                U[i,j] = 1
    return U

def iterative(n, maxIT, method, A, D, L, U, omega = 1):
    h = 1 / (n+1)
    b = [2 * h * h for i in range(1,n+1)]
    full_solution = [i * h * (1 - i * h) for i in range(1,n+1)]
    x = np.zeros([n])
    if method == 'Jacobi':
        M = D
        N = L + U
    if method == 'Gauss-Seidel':
        M = D - L
        N = U
    if method == 'SoR':
        M = (D / omega) - L
        N = ( (1 - omega) / omega ) * D + U
    Mi = np.linalg.inv(M)
    T = np.matmul(Mi, N)
    TOL = 0.000001
    err = 1
    k = 0
    while err > TOL:
        k += 1
        C = np.matmul(Mi,b)
        x = np.matmul(T,x) + C
        err = np.linalg.norm(x - full_solution, np.inf)
    rho = math.exp( (1/k) * math.log10( err / np.linalg.norm(np.zeros([n]) - full_solution, np.inf) ) )
    print("DONE with {}: {} iterations, rho = {}".format(method,k, rho))


    

n = [25, 50, 100, 200]
omega = [1.78486, 1.88402, 1.93968, 1.96922]
maxIT = 2000

for i in range(0,4):
    print("n = {}".format(n[i]))
    A = getA(n[i])
    D = getD(n[i])
    L = getL(n[i])
    U = getU(n[i])
    iterative(n[i], maxIT, 'Jacobi', A, D, L, U)
    iterative(n[i], maxIT, 'Gauss-Seidel', A, D, L, U)
    iterative(n[i], maxIT, 'SoR', A, D, L, U, omega[i])
