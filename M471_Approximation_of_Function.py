# M471_Approximation_of_Function.py
# M471 Dr.Karakashian
# written by Noah Caldwell
# 11/6/18
# Approximates the function f(x) = 1 / (1 + 25x^2) on the interval [-1,1]
# by several approximation techniques including Lagrange polynomial, piecewise
# linear, piecewise cubic Hermite, and cubic spline interpolation with natural B.C.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import solve_banded
from scipy import interpolate

##==========================================================##
#       Horner's algorithm for Lagrange- evaluates function
##==========================================================##
def FCN(z,x,a):
        n = 10
        val = a[n]
        for k in range(n-1,-1,-1):
                val = val*(z-x[k])+a[k]
        return val

##==========================================================##
#       Constants and Arrays
##==========================================================##
h = 0.2
x = [-1.0+(j*h) for j in range(0,11)] # nodes
fx = [1.0/(1.0+(25.0*xi*xi)) for xi in x] # value of f at nodes
plt.ylim(-0.25,1.25) # sets up suitable axes for plotting

##==========================================================##
#       Lagrange polynomial interpolation
##==========================================================##
n = len(x)
F = np.empty((n,n))
for i in range(0,n):
        F[i,0] = fx[i]

# Newton's divided differences
for i in range(1,n):
        for j in range(1,i+1):
                F[i,j] = (F[i,j-1] - F[i-1,j-1]) / (x[i] - x[i-j])

# 11 coefficients down the diagonal
b = [] 
for k in range(0,11):
        b.append(F[k,k])

# Plot functions and writes interpolant and f(x) values to file
def plot_lagrange(x0, x1, nodes, coeff):
        f = open('out1_lagrange_polynomial.txt', 'w')
        z = np.linspace(x0, x1, 250)
        y = 1/(1+(25*z*z))
        p = FCN(z,nodes,coeff)
        maxerror = 0
        for i in range(0,len(z)): # finds max error
                 a = abs(y[i]-p[i])
                 if (a > maxerror):
                         maxerror = a
        plt.scatter(z, p, label="Lagrange Interpolant")
        plt.scatter(z,y, label="f(x)")

        f.write("M471 Lagrange Interpolation: \n")
        f.write("Max error of Lagrange = " + str(maxerror) + "\n\n")
        f.write("x\t\t  interpolant(x)\t  f(x)\t\t  error\n")
        for i in range(0,len(z)):
                f.write("{:.10}\t  {:.10}\t  {:.10}\t  {:.10}\n".format(z[i],p[i],y[i],p[i]-y[i]))
        f.close()

plt.scatter(x, fx, marker="x", color="r")
plot_lagrange(-1,1,x,b)
plt.savefig("Lagrange.png")
plt.clf()


##################################################
#     Piecewise Linear Interpolation  
##################################################

def piecewise_linear(y,fx,x): # y is point, fx is f(x), x is nodes
        for j in range(len(x)):
                if x[j] <= y and y <= x[j+1]: # this is equivalent to using
                                              # the formula j = [5+5x]
                        term1 = fx[j] * (y - x[j+1]) / (x[j] - x[j+1])
                        term2 = fx[j+1] * (y - x[j]) / (x[j+1] - x[j])
                        return term1 + term2

# Plot functions and writes interpolant and f(x) values to file
def plot_linear(x0, x1, fx, x):
        f = open('out2_piecewise_linear.txt', 'w')
        z = np.linspace(x0, x1, 250)
        y = 1/(1+(25*z*z))
        p = []
        for i in range(len(z)):
                p.append(piecewise_linear(z[i],fx,x))
        maxerror = 0
        for i in range(0,len(z)): # finds max error
                 a = abs(y[i]-p[i])
                 if (a > maxerror):
                         maxerror = a
        
        plt.plot(z, p, label="Piecewise Linear Interpolant")
        
        plt.plot(z,y, label="f(x)")

        f.write("M471 Piecewise Linear Interpolant: \n")
        f.write("Max error of Piecewise Linear = " + str(maxerror) + "\n\n")
        f.write("x\t\t  interpolant(x)\t  f(x)\t\t  error\n")
        for i in range(0,len(z)):
                f.write("{:.10}\t  {:.10}\t  {:.10}\t  {:.10}\n".format(z[i],p[i],y[i],p[i]-y[i]))
        f.close()

plot_linear(-1,1,fx,x)
plt.scatter(x, fx, marker="x", color="r")
plt.savefig("Piecewise_linear.png")
plt.clf()


#################################################################
#     Piecewise Cubic Hermite
#################################################################

def phi0(x):
        return 1 - (x*x) + 2*(x*x)*(x-1)
def phi1(x):
        return (x*x) - 2*(x*x)*(x-1)
def phi0hat(x):
        return x - (x*x) + (x*x)*(x-1)
def phi1hat(x):
        return (x*x)*(x-1)
def fprime(x):
        return -(50*x) / (1 + (25*x*x))

def cubic_hermite(y,fx,x):
        for j in range(len(x)):
                if x[j] <= y and y <= x[j+1]: # this is equivalent to using
                                              # the formula j = [5+5x]
                        hj = x[j+1] - x[j]
                        term1 = fx[j] * phi0( (y - x[j]) / hj )
                        term2 = fx[j+1] * phi1( (y - x[j]) / hj )
                        term3 = hj * fprime(x[j]) * phi0hat( (y - x[j]) / hj )
                        term4 = hj * fprime(x[j+1]) * phi1hat( (y - x[j]) / hj )
                        return term1 + term2 + term3 + term4

# Plot functions and writes interpolant and f(x) values to file
def plot_cubic_hermite(x0, x1, fx, x):
        f = open('out3_cubic_hermite.txt', 'w')
        z = np.linspace(x0, x1, 250)
        y = 1/(1+(25*z*z))
        p = []
        for i in range(len(z)):
                p.append(cubic_hermite(z[i],fx,x))
        maxerror = 0
        for i in range(0,len(z)): # finds max error
                 a = abs(y[i]-p[i])
                 if (a > maxerror):
                         maxerror = a
        plt.plot(z, p, label="Cubic Hermite Interpolant")
        plt.plot(z,y, label="f(x)")

        f.write("M471 Cubic Hermite Interpolant: \n")
        f.write("Max error of Cubic Hermite = " + str(maxerror) + "\n\n")
        f.write("x\t\t  interpolant(x)\t  f(x)\t\t  error\n")
        for i in range(0,len(z)):
                f.write("{:.10}\t  {:.10}\t  {:.10}\t  {:.10}\n".format(z[i],p[i],y[i],p[i]-y[i]))
        f.close()

plot_cubic_hermite(-1,1,fx,x)
plt.scatter(x, fx, marker="x", color="r")
plt.savefig("cubic_hermite.png")
plt.clf()


#################################################################
#     Cubic Spline Interpolation
#################################################################

plt.scatter(x, fx, marker="x", color="r")

# plots spline interpolant and outputs data to file
def plot_spline(x0, x1, f0, f1, ppp0, ppp1):
        f = open('out4_cubic_spline.txt', 'w')
        z = np.linspace(x0, x1, 250)
        y = 1/(1+(25*z*z))
        dx = x1-x0

        alpha = ppp1/(6.0*dx)
        beta = -ppp0/(6.0*dx)

        gamma = (-ppp1*dx*dx/6.0 + f1)/dx
        eta = (ppp0*dx*dx/6.0 - f0)/dx

        p = alpha*(z-x0)**3 + beta*(z-x1)**3 + gamma*(z-x0) + eta*(z-x1)
        plt.plot(z, p, label="Cubic Spline Interpolant")
        plt.plot(z,y)

        maxerror = 0
        for i in range(0,len(z)): # finds max error
                a = abs(y[i]-p[i])
                if (a > maxerror):
                        maxerror = a
        
        f.write("M471 Cubic Spline: \n")
        f.write("Max error of Cubic Spline = " + str(maxerror) + "\n\n")
        f.write("x\t\t  interpolant(x)\t  f(x)\t\t  error\n")
        for i in range(0,len(z)):
                f.write("{:.10}\t  {:.10}\t  {:.10}\t  {:.10}\n".format(z[i],p[i],y[i],p[i]-y[i]))
                #f.write(str(z[i]))
        f.close()


xmin = -1
xmax = 1

# coordinates of the data locations
x = np.linspace(xmin, xmax, n+1)
dx = x[1] - x[0]

f = np.empty(12)
for i in range(0,len(fx)):
        f[i] = fx[i]

# we are solving for n-1 unknowns
# setup the righthand side of our matrix equation
b = np.zeros(n+1)

# b_i = (6/dx) * (f_{i-1} - 2 f_i + f_{i+1})
# here we do this with slice notation to fill the
# inner n-1 slots of b
b[1:n] = (6.0/dx)*(f[0:n-1] - 2.0*f[1:n] + f[2:n+1])

# we only care about the inner n-1 quantities
b = b[1:n]

# the matrix A is tridiagonal.  Create 3 arrays which will represent
# the diagonal (d), the upper diagonal (u), and the lower diagnonal
# (l).  l and u will have 1 less element.  For u, we will pad this at
# the beginning and for l we will pad at the end.

u = np.zeros(n-1)
d = np.zeros(n-1)
l = np.zeros(n-1)

d[:] = 4.0*dx

u[:] = dx
u[0] = 0.0

l[:] = dx
l[n-2] = 0.0

# create a banded matrix -- this doesn't store every element -- just
# the diagonal and one above and below
A = np.matrix([u,d,l])
# solve Ax = b using the scipy banded solver -- the (1,1) here means
# that there is one diagonal above the main diagonal, and one below
xsol = solve_banded((1,1), A, b)
# x now hold all the second derivatives for points 1 to n-1.  Natural
# boundary conditions set p'' = 0 at i = 0 and n
# ppp will be our array of second derivatives
ppp = np.insert(xsol, 0, 0)  # insert before the first element
ppp = np.insert(ppp, n, 0)   # insert at the end


# plot the splines
for i in range(n):
    # working on interval [i,i+1]
    pppi = ppp[i]
    pppip1 = ppp[i+1]

    fi = f[i]
    fip1 = f[i+1]

    xi = x[i]
    xip1 = x[i+1]

    plot_spline(xi, xip1, fi, fip1, pppi, pppip1)

plt.savefig("spline.png")

