# M471_Approximate_ODEs.py
# M471 Dr.Karakashian
# written by Noah Caldwell
# 11/30/18
# Implements several methods for approximation of an IVP for ODEs
# and experimentally calculates the rate of convergence.
# Includes Taylor series method, RK4, and Two-Step Method.

import math
import time # I calculated timing just for fun

def exactSol(t): # exact solution
    return t / (1 + math.log(t))

def FCN(tn,Yn): # f
    return (Yn/tn) - ((Yn*Yn) / (tn*tn))
def FCNy(tn,Yn): # f_y
    return (1/tn) - (2 * Yn) / (tn*tn)
def FCNt(tn,Yn): # f_t
    return -(Yn / (tn*tn)) + (2*(Yn*Yn) / (tn*tn*tn))

def taylorMethod(t0,y0,tend,Nsteps):
    h = (tend - t0) / Nsteps
    Yn = y0
    tn = t0
    ERRmax = 0
    print("Taylor series method:")
    # print("tn\t\t" + "Yn\t\t" + "Yexactn\t\t" + "ERRn\t\t" + "ERRmax")
    for i in range(0,Nsteps):
        Yn = Yn + h * FCN(tn,Yn) + ((h*h) / 2) * (FCNt(tn,Yn) + FCNy(tn,Yn) * FCN(tn,Yn))
        tn = tn + h
        Yexactn = exactSol(tn)
        ERRn = abs(Yexactn - Yn)
        ERRmax = max(ERRmax, ERRn)
        # print(str(tn)+"\t\t"+str(round(Yn,i))+"\t\t"+str(round(Yexactn,i))+"\t\t"+str(round(ERRn,i))+"\t\t"+str(round(ERRmax,i)))
    print("DONE: at tn="+str(tn)+", Yn="+str(Yn) + ", with Error="+str(ERRmax))

def RK4(t0,y0,tend,Nsteps):
    h = (tend - t0) / Nsteps
    Yn = y0
    tn = t0
    ERRmax = 0
    print("Runge-Kutta method:")
    # print("tn\t\t" + "Yn\t\t" + "Yexactn\t\t" + "ERRn\t\t" + "ERRmax")
    for i in range(0,Nsteps+1):
        k1 = h * FCN(tn,Yn)
        k2 = h * FCN(tn + (0.5*h), Yn + (0.5*k1))
        k3 = h * FCN(tn + (0.5*h), Yn + (0.5*k2))
        k4 = h * FCN(tn + h, Yn + k3)
        Yn = Yn + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        tn = tn + h
        Yexactn = exactSol(tn)
        ERRn = abs(Yexactn - Yn)
        ERRmax = max(ERRmax, ERRn)
        if i == 0:
            YnRK = Yn
        # print(str(tn)+"\t\t"+str(round(Yn,i))+"\t\t"+str(round(Yexactn,i))+"\t\t"+str(round(ERRn,i))+"\t\t"+str(round(ERRmax,i)))
    print("DONE: at tn="+str(tn)+", Yn="+str(Yn) + ", with Error="+str(ERRmax))
    return YnRK # returns y1 for method (3)

def twoStepMethod(t0,y0,tend,Nsteps,YnRK):
    h = (tend - t0) / Nsteps
    Yn1 = y0 # Yn-1
    tn1 = t0 # tn-1
    Yn = YnRK # Yn
    tn = t0 + h # tn
    ERRmax = 0
    print("Two-step Method:")
    # print("tn\t\t" + "Yn\t\t" + "Yexactn\t\t" + "ERRn\t\t" + "ERRmax")
    for i in range(1,Nsteps+1):
        tempYn = Yn
        Yn = Yn + (h/2)*(3*FCN(tn,Yn) - FCN(tn1,Yn1)) # Yn
        Yn1 = tempYn # Yn-1
        tn1 = tn1 + h # tn-1
        tn = tn + h # tn
        Yexactn = exactSol(tn)
        ERRn = abs(Yexactn - Yn)
        ERRmax = max(ERRmax, ERRn)
        # print(str(tn)+"\t\t"+str(round(Yn,i))+"\t\t"+str(round(Yexactn,i))+"\t\t"+str(round(ERRn,i))+"\t\t"+str(round(ERRmax,i)))
    print("DONE: at tn="+str(tn)+", Yn="+str(Yn) + ", with Error="+str(ERRmax))


# Running functions with varying h's
t0 = 1
y0 = 1
tend = 2
Nsteps = 25 # h = 1/25
print("h = {}".format(Nsteps))
taylorMethod(t0,y0,tend,Nsteps)
YnRK = RK4(t0,y0,tend,Nsteps)
twoStepMethod(t0,y0,tend,Nsteps,YnRK)

print()
print()
Nsteps = 50 # h = 1/50
print("h = {}".format(Nsteps))
taylorMethod(t0,y0,tend,Nsteps)
YnRK = RK4(t0,y0,tend,Nsteps)
twoStepMethod(t0,y0,tend,Nsteps,YnRK)

print()
print()
Nsteps = 100 # h = 1/100
print("h = {}".format(Nsteps))
taylorMethod(t0,y0,tend,Nsteps)
YnRK = RK4(t0,y0,tend,Nsteps)
twoStepMethod(t0,y0,tend,Nsteps,YnRK)

print()
print()
Nsteps = 200 # h = 1/200
print("h = {}".format(Nsteps))
taylorMethod(t0,y0,tend,Nsteps)
YnRK = RK4(t0,y0,tend,Nsteps)
twoStepMethod(t0,y0,tend,Nsteps,YnRK)

print()
print()
Nsteps = 400 # h = 1/400
print("h = {}".format(Nsteps))
start1 = time.time() # timing, just for fun
taylorMethod(t0,y0,tend,Nsteps)
end1 = time.time()
start2 = time.time()
YnRK = RK4(t0,y0,tend,Nsteps)
end2 = time.time()
start3 = time.time()
twoStepMethod(t0,y0,tend,Nsteps,YnRK)
end3 = time.time()

print()
print()
print("Timing, just for kicks:")
# elapsed time during function call with h=400
print("taylor method: {}".format(end1-start1))
print("RK4: {}".format(end2-start2))
print("2-step: {}".format(end3-start3))


# calculating rate
"""
e1 = 0.00022060714937266468
e2 = 5.347888799578371e-05
e3 = 1.3164891212946728e-05
e4 = 3.265893901538419e-06
e5 = 8.133247488828488e-07
print(math.log(e1/e2) / math.log(0.5))
print(math.log(e2/e3) / math.log(0.5))
print(math.log(e3/e4) / math.log(0.5))
print(math.log(e4/e5) / math.log(0.5))
"""


"""
taylorMethod(t0,y0,tend,Nsteps)
YnRK = RK4(t0,y0,tend,Nsteps)
twoStepMethod(t0,y0,tend,Nsteps,YnRK)

print()
print()
"""
