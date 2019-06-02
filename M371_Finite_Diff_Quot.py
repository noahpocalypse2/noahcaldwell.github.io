# M371_Finite_Diff_Quot.py
# M371 Dr.Alexiades
# written by Noah Caldwell
# 2/26/18
# Approximates the derivative of a function using both
# the forward and centered finite difference quotient.

import math

def ffdq(x,h): # centered finite difference quotient
    fprime = (FCN(x+h)-FCN(x))/(h)
    return fprime

def cfdq(x,h): # centered finite difference quotient
    fprime = (FCN(x+h)-FCN(x-h))/(2*h)
    return fprime

def FCN(x):
    return math.sin(x)

x = float(input("Define x: "))
N = int(input("DefineN: "))

def lab4(x,N):
    f = open('out.txt', 'w') 
    print("k\t" + "h\t" + "ERRn")
    for k in range(5,N+1):
        h = 1/(2**k)
        ERRn = abs(math.cos(x)-cfdq(x,h))
        print(str(k) + " " + str(h) + " " + str(ERRn))
        f.write(str(k)+" "+str(math.log(h))+" "+str(math.log(ERRn))+"\n")

lab4(x,N)

# Apparently, the io functions above only work when used in a function.
# Otherwise I wouldn't have bothered with defining lab4(x,N)
