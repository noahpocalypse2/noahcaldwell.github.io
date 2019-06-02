# M475_Explicit_Finite_Volume_scheme.py
# M475 Dr.Alexiades
# written by Noah Caldwell
# 3/27/18
# Implement the explicit numerical scheme for 1-D diffusion of a
# concentration u(x,t) in an interval [a, b], during time [0, tend].


def mesh(a,b,dx,x,M): # constructs mesh: x(i), i=0,...,M+1
    x[0] = a
    x[1] = dx/2
    x[M+1] = b
    for i in range (1, M+1):
        x[i] = x[1]+(i-1)*dx
    

def init(u,M): # set initial values for u(i), i=0,...,M+1
    for i in range(0,M+2):
        u[i] = 0
        if (x[i]>=1 and x[i]<=2):
            u[i]=5

def PDE(u,M,dt,dx,F): # update U(i), i=0,...,M+1
    for i in range(1,M+1):
        u[i]=u[i]+(dt/dx)*(F[i]-F[i+1])
    u[0] = D*u[1]/((dx/2)*V+D)
    if (D==0): u[M+1]=u[M]
    if (D!=0): u[M+1] = (u[M]*((dx/2)*V+D))/D
    
def flux(u,M,F,b,D,time): # compute fluxes F(i), i=1,...,M+1\
    F[1] = 0
    F[M+1] = 0
    for i in range(2,M+1): # total flux equals sum of diffusion and advection flux
        F[i] = V*u[i-1]-D*(u[i]-u[i-1])/dx
        
def compare(ERR): # compute exact solution and error
    for i in range(0,M+2):
        uExact[i] = (1.0 / math.sqrt(4.0*math.pi*D)) * math.exp((-1.0*x[i]*x[i])/(4.0*D*time))
        ERRi = abs(u[i] - uExact[i])
        ERR = max(ERRi,ERR)
    return ERR

def trapzrule(M,dx,u):
    sum = 0 #reset to zero every time you run the function
    sum = (u[0]+3*(u[1]+u[M])+u[M+1])/4.0
    for i in range(2,M):
        sum = sum + u[i]
    return dx*sum

def output(): # print profile at particular time (for plotting)
    print("# Profile at time: " + str(time) + ", nsteps = " + str(nsteps))
    print("Trapezoidal rule area calculation: " + str(trapzrule(M,dx,u)))
    print("x[i] \t u[i]")
    for i in range (0,M+2):
        print(str(x[i]) + ",   " + str(u[i]))
    print()
    

# ==========================================
# declare variables and read/write
# ==========================================
f = open('dat.txt', 'r')
g = open('lab8output.txt', 'w')

numbers = f.readline()
numbers = str.split(numbers)

MM = float(numbers[0]) # 
tend = float(numbers[1]) # t at which program ends
dtout = float(numbers[2]) # 
factor = float(numbers[3]) # 
D = float(numbers[4]) # 
a = float(numbers[5]) # beginning of rod
b = float(numbers[6]) # end of rod
V = 5.0 # velocity

t0 = 0.0 # initial time

# derived variables
dx = 1.0 / MM
M = int((b-a) * MM)

# setting up arrays
x = [0]*(M+2) # python lists are weird. this initializes to specific length
u = [0]*(M+2)
F = [1]*(M+2)

# x is the discretized location on the rod.
# u is the value of equation, the amount of heat.

mesh(a,b,dx,x,M)
# ==========================================
# ............ set the timestep ............
# ==========================================
if (D>0 and V==0): dtEXPL = (dx*dx) / (2*D) # max timestep for stability in diffusion
if (D==0 and V>0): dtEXPL = dx/V # max timestep for stability in advection
if (D>0 and V>0): dtEXPL = (dx*dx) / (2*D + (dx*V))       # max timestep for stability in advection-diffusion


dt = factor * dtEXPL	        # factor<1 increases stability
Nend = int((tend-t0)/dt) + 1	# of timesteps
# ==========================================
# .............. initialize ................
# ==========================================
init(u,M)
nsteps = 0
time = 0.0
tout = max(dtout,dt)
#output()    # to print initial profile
for i in range(0,M+2):
    g.write(str(x[i]) + "\t" + str(u[i]) + "\n") # prints to file# ==========================================
# ........... begin time-stepping ..........
# ==========================================
for nsteps in range(1, Nend+1):
    time = nsteps*dt
    flux(u,M,F,b,D,time)
    PDE(u,M,dt,dx,F)
    
    if (time >= tout):
        #output()
        print("Trapezoidal rule area calculation: " + str(trapzrule(M,dx,u)))
        for i in range(0,M+2):
            g.write(str(x[i]) + "\t" + str(u[i]) + "\n") # prints to file
        tout = tout + dtout 
    if (time >= tend):
        print("DONE, at time="+str(time)+", nsteps=" + str(nsteps))
# ==========================================
# .......... end of time-stepping ..........
# ==========================================
#print("... out of timesteps: need bigger Nend")
f.close()
g.close()
# END

