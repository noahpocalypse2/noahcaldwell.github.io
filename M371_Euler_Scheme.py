# M371_Euler_Scheme.py
# M371 Dr. Alexiades
# written by Noah Caldwell
# 4/23/18
# Implements an Euler scheme to calculate the number of moles KOH present in the reaction
# 2K_2 Cr_2 O + 2H_2 ) + 3S ---> 4KOH + 2Cr_2 O_3 + 3SO_2
# The reaction is described by the ODE y' = k (m1 - y/2)^2 (m2 - y/2)2(m3 - 3y/4)^3
# where y(t) is number of moles of KHO present at time t.

import math

def EulerScheme(t0,y0,tend,Nsteps):
    Yn = y0
    tn = t0
    dt = (tend - t0) / Nsteps # time step
    ERRmax = 0 # giving this an initial value
    #print("tn\t\t" + "Yn\t\t" + "Yexactn\t\t" + "ERRn\t\t" + "ERRmax")
    for n in range(1,Nsteps+1):
        Yn = Yn + dt * FCN(tn,Yn)   # These top two steps perform the actual algorithm.
        tn = t0 + n * dt            # Yn is what we want to calculate.
        #Yexactn = math.sqrt(1.0-tn*tn)
        #ERRn = abs(Yexactn - Yn)
        #ERRmax = max(ERRmax, ERRn)
        #print(str(tn)+"\t\t"+str(Yn)+"\t\t"+str(Yexactn)+"\t\t"+str(ERRn)+"\t\t"+str(ERRmax))
    print("DONE: at tn="+str(tn)+", Yn="+str(Yn)) #+", with Error="+str(ERRmax))


def FCN(tn,Yn):
    return k*(m1-(Yn/2.0))*(m1-(Yn/2.0))*(m2-(Yn/2.0))*(m2-(Yn/2.0))*(m3-(3*Yn/4.0))*(m3-(3*Yn/4.0))*(m3-(3*Yn/4.0)) # the function is y'=-t/y

m1 = 2000
m2 = 2000
m3 = 3000
k = 6.22e-19
t0 = 0 #float(input("Define t0: "))
y0 = 0 #float(input("Define y0: "))
tend = 0.5 #float(input("Define tend: "))
Nsteps = 1000000 #int(input("Define Nsteps: "))
EulerScheme(t0,y0,tend,Nsteps)
