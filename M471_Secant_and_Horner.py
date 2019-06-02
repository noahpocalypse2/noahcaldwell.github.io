# M471_Secant_and_Horner.py
# M471 Dr.Karakashian
# written by Noah Caldwell
# 9/29/18
# Implements the Secant algorithm, calculating
# f(x) and f'(x) with Horner's algorithm.

import math

##==========================================================##
#       Secant algorithm
##==========================================================##
def Secant(x0,x1):
    print("Secant method:")
    y0,y1 = FCN(x0),FCN(x1)
    print("n\txn\ten\tratio")
    n = 0
    alpha = 0.5*(1+math.sqrt(5))
    r = 0.2323529647499171
    while (math.fabs(x1-x0) > 0.000000001): # 10^-10
        x = x1-y1*(x1-x0)/(y1-y0)
        y = FCN(x)
        x0,x1 = x1,x
        y0,y1 = y1,y
        en = math.fabs(x-r)
        ratio = math.fabs(r-x1) / ((math.fabs(r-x0))**alpha)
        prevx = x
        print("{}\t{}\t{}\t{}".format(n,x,en,ratio))
        n += 1
    print("DONE: root="+str(x)+", F(root)="+str(y)+", in "+str(n)+" iters")
    print()

##==========================================================##
#       Horner's algorithm (basic and improved) below
##==========================================================##
def FCN(x): # basic Horner's algorithm, returns f(x)
    a = [600,-550,200,-20,-1]
    a.reverse()
    p = a[-1]
    i = len(a)-2
    while i>= 0:
        p = p*x+a[i]
        i -= 1
    return p
def FCND(x): # improved Horner's algorithm: returns f(x) and f'(x), aka p and q
    a = [600,-550,200,-20,-1]
    a.reverse()
    b = []
    p = a[-1]
    i = len(a)-2
    while i>= 0:
        b.append(p)
        p = p*x+a[i]
        i -= 1
    b.reverse()
    q = b[-1]
    i = len(b)-2
    while i>= 0:
        q = q*x+b[i]
        i -= 1
    return p,q

##==========================================================##
#       Function called below
##==========================================================##

x0 = 0.1
x1 = 0.12
Secant(x0,x1)


