# M472_Proj1_Hilbert.py
# M472 Dr. Karakashian
# written by Noah Caldwell
# 3/14/19
# Math 472

# This program creates Hilbert matrices (specifically, H(8)), calculates the 
# inverse via formula, and factorizes Hilbert matrices with Cholesky and QR.
# The inverse of Hilbert is calculated by solving Ax=b with Cholesky and QR,
# and the infinity norm of the difference between the approximation and the
# calculated value is calculated.

import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=300,precision=2)

def Hilbert(n):
    # Creates n x n Hilbert matrix with formula 1)
    mat = np.zeros([n,n])
    for i in range(1,n+1):
        for j in range(1,n+1):
            mat[i-1,j-1] = 1.0 / (i+j-1)
    return mat

def HilbertInverse(n):
    mat = np.zeros([n,n])
    for i in range(1,n+1):
        for j in range(1,n+1):
            mat[i-1,j-1] = ((-1)**(i+j)) * (i+j-1) * sp.binom(n+i-1,n-j) * sp.binom(n+j-1,n-i) * (sp.binom(i+j-2,i-1)**2)
    return mat

def Choleski(A):
    n = len(A[1])
    L = np.zeros([n,n])
    n = n-1 # so that A[n] refers to the actual last value, since arrays start at 0
    L[0,0] = float(np.sqrt(A[0,0]))
    for i in range(1,n+1):
        L[i,0] = A[i,0] / L[0,0]
    for i in range(1,n): # n-1?
        L[i,i] = np.sqrt(A[i,i] - sum([L[i,K] * L[i,K] for K in range(0,n)]) )
        for j in range(i+1, n+1):
            L[j,i] = (A[j,i] - sum( [L[j,K] * L[i,K] for K in range(0,i)]) ) / L[i,i]
    L[n,n] = np.sqrt(A[n,n] - sum( [L[n,K] * L[n,K] for K in range(0,n)]) )
    return L

def back_substitution(A, b):
    # calculates the solution x of an upper triangular matrix
    n = len(A[1])
    x = np.zeros((n,1))
    n = n-1 # so that A[n] refers to the actual last value, since arrays start at 0
    x[n] = b[n] / A[n, n]
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n+1):
            s = s - A[i,j] * x[j]
        x[i] = s / A[i,i]
    return x

def forward_substitution(A, b):
    # calculates the solution y of a lower triangular matrix
    n = len(A[1])
    y = np.zeros([n,1])
    for i in range(n):
        s = 0
        for j in range(i):
            s = s + (L[i, j]*y[j])
        y[i] = (b[i] - s) / L[i, i]
    return y


print("Part 2: K-infinity (H(8))")
cond_numbers_hilbert = np.array([])
Kinf_list = np.array([])
for i in range(1,7):
    hilbert = Hilbert(i*5)
    hilbert_inverse = HilbertInverse(i*5)
    x1 = np.linalg.norm(hilbert,np.inf)
    x2 = np.linalg.norm(hilbert_inverse,np.inf)
    cond_numbers_hilbert = np.append(cond_numbers_hilbert, x1*x2)
Kinf_list = []
for i in range(0,len(cond_numbers_hilbert)):
    Kinf_list.append((((1+np.sqrt(2))**(4*(i+1)*5)) / np.sqrt((i+1)*5)))
for i in range(0,6):
    print("n = {}, \nK-inf(H(8)) * K-inf(H(8)^-1) = {}, \n      K-inf from formula (2) = {}".format((i+1)*5, cond_numbers_hilbert[i],Kinf_list[i]))

print("\nPart 3: Choleski Factorization: see functions\n")
# See functions above: Choleski(A), back_substitution(A, b), forward_substitution(A, b)
hilbert = Hilbert(8)
print("H(8):")
print(hilbert)

print("\nPart 4: H(8)^-1:")
hilbert_inverse = HilbertInverse(8)
print(hilbert_inverse)

print("\nPart 5: Inverse Choleski:")
L = Choleski(hilbert)
print("\nL:")
print(L)
b = np.zeros([8])
b[0] = 1
y = forward_substitution(L, b)
choleski_inverse = back_substitution(L.T,y)
b[0] = 0
for i in range(1,8):
    b[i] = 1
    y = forward_substitution(L,b)
    invcol = back_substitution(L.T, y)
    b[i] = 0
    choleski_inverse = np.append(choleski_inverse,invcol,1)
print("\nInverse of Choleski approximation:")
print(choleski_inverse)

print("\nPart 6: Infinity norm of Part 4 - Part 5")
part6 = np.linalg.norm(hilbert_inverse - choleski_inverse, np.inf)
print(part6)

print("\nPart 7: QR factorization of H(8)^-1")
q,r = np.linalg.qr(Hilbert(8))
b[0] = 1
QRinverse = back_substitution(r, np.matmul(q.T,b))
b[0] = 0
for i in range(1,8):
    b[i] = 1
    invcol = back_substitution(r, np.matmul(q.T,b))
    b[i] = 0
    QRinverse = np.append(QRinverse,invcol,1)
print(QRinverse)

print("\nPart 8: Infinity norm of Part 4 - Part 7")
part8 = np.linalg.norm(hilbert_inverse - QRinverse, np.inf)
print(part8)
