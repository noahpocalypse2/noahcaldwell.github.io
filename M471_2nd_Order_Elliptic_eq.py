# M471_2nd_Order_Elliptic_eq.py
# M471 Dr.Alexiades
# written by Noah Caldwell
# 11/6/18
# Approximates the solution of a second-order elliptic differential equation.
# Uses the finite difference method which is based on the centered finite
# difference approximation of a second-derivative:
# u''(x) ~~ ( u(x-h) - 2u(x) + u(x+h) ) / h^2

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

pi = math.pi

def u(x):
    return math.sin(pi*x)
def upp(x):
    return pi*pi*math.sin(pi*x)

h1 = 1/25
h2 = 1/50
h3 = 1/100
h4 = 1/200
N1 = 24 
N2 = 49 # h2**-1 - 1
N3 = 99 # h3**-1 - 1
N4 = 199 # h4**-1 - 1
a = 0
b = 1
alpha = 0
Beta = 0

for i in range(0,4):
    if i == 0:
        h = h1
        N = N1
        name = 'h1'
        print("h = 1/25: ")
    elif i == 1:
        h = h2
        N = N2
        name = 'h2'
        print("h = 1/50:")
    elif i == 2:
        h = h3
        N = N3
        name = 'h3'
        print("h = 1/100:")
    elif i == 3:
        h = h4
        N = N4
        name = 'h4'
        print("h = 1/200:")
    uvec = np.zeros(N+3)
    uvec[0] = alpha
    uvec[N+2] = Beta

    # form the matrix
    Bmatrix = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        for k in range(0,N+1):
            if j == k:
                Bmatrix[j,k] = 2
            if k - j == 1 or j - k == 1:
                Bmatrix[j,k] = -1

    # form right hand side
    rms = np.zeros(N+1)
    x = np.zeros(N+2)
    for j in range(0,N+2):
        x[j] = a + (j*h)
    for j in range(0,N+1):
        rms[j] = h**2 * upp(x[j])
    m = np.zeros(N)
    m = scipy.linalg.solve(Bmatrix, rms)

    for j in range(1,N+2):
        uvec[j] = m[j-1]
    fx = np.zeros(N+1)
    fdubprime = np.zeros(N+1)
    for j in range(1,N+2):
        fx[j-1] = -(uvec[j-1] - (2*uvec[j]) + uvec[j+1]) / (h**2)
        fdubprime[j-1] = upp(x[j-1])
    
    z = np.linspace(a, b, N)
    y = []
    for i in range(len(z)):
        y.append(u(z[i]))
    p = []
    for i in range(len(z)):
        p.append(uvec[i])
    plt.plot(z,p, 'ko', markersize=2)
    plt.plot(z,y, markersize=2)
    plt.savefig(name + ".png")
    plt.clf()

    err = 0
    maxerr = 0
    for j in range(1,N):
        temp = abs(u(x[j]) - uvec[j])
        if temp > maxerr:
            maxerr = temp
        err = err + ((temp)**2)
    err = ((err * (1/N)) ** (1/2))
    print("Error: ", err)
    print("Max error: ", maxerr)
    print()
