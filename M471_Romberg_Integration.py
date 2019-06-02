# M471_Romberg_Integration.py
# M471 Dr.Karakashian
# written by Noah Caldwell
# 11/20/18
# Implements Romberg Integration. This consists of applying
# Richardson extrapolation to the composite trapezoidal rule.

import math
import numpy as np
np.set_printoptions(linewidth=200,precision=15)

### Constants and arrays
pi = math.pi
fsol = 2 / pi
a = 0.0
b = 1.0
N = 7
F = np.zeros((N,N))
error = np.zeros((N,N))

def f(x):
    return math.sin(pi*x)

### find R(0,0) with basic trapezoidal rule
F[0,0] = ( (b-a) / 2) * (f(a) + f(b))

### Composite trap rule:
### find R(n,0) for n = 1,...,6
for n in range(1,N): # 6+1 because Python loops do not evaluate the last value
    h = (b - a) / (2**n)
    #float(h)
    term1 = 0.5 * F[n-1,0]
    term2 = h * sum([f(a + ((2.0*j) - 1.0) * h) for j in range(1,2**(n-1)+1)])
    F[n,0] = term1 + term2


### Now, apply Richardson extrapolation
for n in range(1,N):
    for m in range(1,n+1):
        term1 = F[n,m-1]
        term2 = ( F[n,m-1] - F[n-1,m-1] ) / ( (4**m) - 1)
        F[n,m] = term1 + term2

### Calculating error term: E(n,m) = |R(n,m) - 2/pi|, 0 <= m <= n <= 6
for n in range(0,N):
    for m in range(0,n+1):
        error[n,m] = abs( F[n,m] - (2/pi) )

print(F)
print()
print(error)
