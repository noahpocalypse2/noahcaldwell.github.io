# M475_Proj2_Ostwald.py
# M475 Dr.Alexiades
# written by Noah Caldwell
# 3/29/18
# Verify numerically the theoretical predictions of the Ostwald
# ripening model with two crystals of different initial sizes.
# Utilizes Newton-Raphson method and Euler scheme.


def Newton1D(x0,TOL,maxIT): #performs the iterations
    xn = x0
    dx = 1000.0
    print("n\t"+"xn\t\t\t"+"Fn") # labels for values
    for n in range(1,maxIT+1):
        Fn, DFn = FCN(xn) # runs through Newton's method
        #print(str(n)+"\t"+str(xn)+"\t\t\t"+str(Fn)) # prints out current iteration values
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
    xstar = 0.09
    c1 = c0 + (mu*xstar*xstar*xstar)
    
    Fn = (mu*xn*xn*xn)+(cstar*math.exp(gamma/xn)-c1)
    DFn = (3*mu*xn*xn)-((gamma*cstar*math.expm1(gamma/xn))/(xn*xn))
    return Fn, DFn


while(True):
    x0 = float(input("Define x0: ")) #initial guess for root
    maxIT = int(input("Define maxIT: ")) #maximum number iterations to be performed
    TOL = 0.000000000000001 #float(input("Define TOL: ")) #tolerance for testing convergence
    Newton1D(x0,TOL,maxIT)




def EulerScheme(t0,y10,y20,tend,Nsteps):
    f = open('output', 'w')
    Y1n = y10 # Y1n = size of crystal 1
    Y2n = y20 # Y2n = size of crystal 2
    tn = t0 # initial time
    dt = (tend - t0) / Nsteps # time step
    print("tn\t" + "Y1n\t" + "Y2n\t" + "ERRn")
    for n in range(1,Nsteps+1): # must go (1, N+1) due to Python for loops starting at 0
        if (Y1n > 0): Y1n = Y1n + (dt * FCN(tn,Y1n,Y1n,Y2n))
        Y2n = Y2n + (dt * FCN(tn,Y2n,Y1n,Y2n))
        tn = t0 + (n * dt)
        f.write(str(tn) + "\t" + str(Y2n))
        if (Y1n>=0):
            f.write("\t" + str(Y1n) + "\n")
        if (Y1n<0): Y1n=0
        #print(str(tn)+"\t"+str(Y1n)+"\t"+str(Y2n))
        if (Y2n<=0): break # or Y2n<=0): break # if either crystal fully dissolves, break
    print("DONE: at tn="+str(tn)+", Y1n="+str(Y1n)+", Y2n="+str(Y2n))


def FCN(tn,Yn,Y1n,Y2n):
    mu = 1.e-3 #conversion variable: converts volume to concentration
    cstar = 7.52e-7
    gamma = 4.e-3
    c0 = 1.05*cstar
    k = 5.e7
    x1star = 0.05 # initial size of crystal 1
    x2star = 0.09 # initial size of crystal 2
    
    ct = c0 + (mu*x1star*x1star*x1star) + (mu*x2star*x2star*x2star) - (mu*Y1n*Y1n*Y1n + mu*Y2n*Y2n*Y2n) # concentration
    Yn = k*(ct-(cstar*math.exp(gamma/Yn)))
    
    return Yn


t0 = 0 #float(input("Define t0: "))
y10 = 0.05 #float(input("Define y10: "))
y20 = 0.09 #float(input("Define y20: "))
while(True):
    tend = float(input("Define tend: "))
    Nsteps = int(input("Define Nsteps: "))
    EulerScheme(t0,y10,y20,tend,Nsteps)

