# Bisection_Method.py
# M371 Dr.Alexiades
# written by Noah Caldwell
# 1/31/18
# Implements the Bisection method to calculate roots of
# the function f(x)=x^3+2x^2+10x-20.

def Bisect(a,b,TOL,maxIT):
    Fa = FCN(a)
    Fb = FCN(b)
    print("n\t" + "xn\t\t\t" + "Fn\t\t\t" + "ERRn") # labels for values
    for i in range (0,maxIT):
        r = (a+b)/2.0
        Fr = FCN(r) # runs through Bisection method
        err = abs(b-a)/2.0
        if (Fa*Fr<0):
            b = r
            Fb = Fr
        else:
            a = r
            Fa = Fr
        print(str(i+1) + "\t" + str(r) + "\t" + str(Fr) + "\t" + str(err)) # current iter values
        if (abs(err) <= TOL):
            if (abs(Fr) <= TOL):
                print("DONE: root=" + str(r) + " residual = " + str(Fr) + " in " + str(i+1) + " iters")
                break
            else:
                print("STUCK: error < TOL but residual=" + str(Fr) +" > TOL")
                break
        if(i == maxIT-1):
            print("BAD: reached maxIT")

def FCN(x): # recalculates Fn=F(xn)
    Fn = x*x*x+2*x*x+10*x-20
    return Fn

a = float(input("Define a: "))
b = float(input("Define b: "))
TOL = float(input("Define tolerance: ")) # tolerance for testing convergence
maxIT= int(input("Define number of maximum iterations: "))
Bisect(a,b,TOL,maxIT)
