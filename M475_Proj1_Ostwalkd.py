# M475_Proj1_Ostwald.py
# M475 Dr.Alexiades
# written by Noah Caldwell
# 2/15/18
# Verify numerically the theoretical predictions of the
# Ostwald ripening model with single-size crystals.
# Utilizes Newton-Raphson method and Euler scheme.

import math

def Newton1D(x0,TOL,maxIT): #performs the iterations
    xn = x0
    dx = 1000.0
    print("n\t"+"xn\t\t\t"+"Fn") # labels for values
    for n in range(1,maxIT+1):
        Fn, DFn = FCN(xn) # runs through Newton's method
        print(str(n)+"\t"+str(xn)+"\t\t\t"+str(Fn)) # prints out current iteration values
        if(math.fabs(dx) < TOL):
            if(math.fabs(Fn) < TOL):
                print("DONE: root="+str(xn)+", F="+str(Fn)+", in "+str(n)+" iters")
                break
            else:
                print("STUCK: dx < TOL BUT residual ="+str(Fn)+" > TOL")
                break
        dx = -Fn/DFn #take Newton step
        xn = xn+dx
    if (n == maxIT): print("BAD: reached maxIT")


def FCN(xn): # recalculates Fn=F(xn) and DFn=F'(xn)
    
    mu = 1.e-3       #various variables and constants
    cstar = 7.52e-7
    gamma = 4.e-3
    c0 = 1.05*cstar
    xstar = 0.05
    c1 = c0 + (mu*xstar*xstar*xstar)
    
    Fn = (mu*xn*xn*xn)+(cstar*math.exp(gamma/xn)-c1)
    DFn = (3*mu*xn*xn)-((gamma*cstar*math.expm1(gamma/xn))/(xn*xn))
    return Fn, DFn

"""
x0 = float(input("Define x0: ")) #initial guess for root
maxIT = int(input("Define maxIT: ")) #maximum number iterations to be performed
TOL = float(input("Define TOL: ")) #tolerance for testing convergence
Newton1D(x0,TOL,maxIT)
"""


def EulerScheme(t0,y0,tend,Nsteps):
    f = open('output', 'w')
    Yn = y0
    tn = t0
    dt = (tend - t0) / Nsteps # time step
    print("tn\t" + "Yn\t" + "Yexactn" + "ERRn")
    for n in range(1,Nsteps+1): # must go (1, N+1) due to Python for loops starting at 0
        Yn = Yn + (dt * FCN(tn,Yn))
        tn = t0 + (n * dt)
        f.write(str(tn) + " " + str(Yn) + "\n")
        if (Yn<=0): break
        #print(str(tn)+"\t"+str(Yn)+"\t")
    print("DONE: at tn="+str(tn)+", Yn="+str(Yn))


def FCN(tn,Yn):
    mu = 1e-5 #conversion variable: converts volume to concentration; 1e-3 for case 1 and 2
    cstar = 7.52e-7
    gamma = 4e-3
    c0 = 1.05*cstar
    k = 5e7
    xstar = 0.08 #changes for each case
    
    ct = c0 + (mu*xstar*xstar*xstar) - (mu*Yn*Yn*Yn)
    Yn = k*(ct-(cstar*math.exp(gamma/Yn)))
    
    return Yn # k*(c1 - mu*(xstar*xstar*xstar)) - (cstar*math.exp(gamma/xstar))


t0 = float(input("Define t0: "))
y0 = float(input("Define y0: "))
tend = float(input("Define tend: "))
Nsteps = int(input("Define Nsteps: "))
EulerScheme(t0,y0,tend,Nsteps)
