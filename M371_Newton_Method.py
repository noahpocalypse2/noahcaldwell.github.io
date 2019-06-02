# M371_Newton_Method.py
# M371 Dr.Alexiades
# written by Noah Caldwell
# 1/31/18
# Implements Newton's Method to calculate roots of
# the function f(x)=x^3+2x^2+10x-20.

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
    Fn = xn-math.tan(xn)
    DFn = 1- math.acos(xn)*math.acos(xn)
    return Fn, DFn


x0 = float(input("Define x0: ")) #initial guess for root
maxIT = int(input("Define maxIT: ")) #maximum number iterations to be performed
TOL = float(input("Define TOL: ")) #tolerance for testing convergence
Newton1D(x0,TOL,maxIT)
