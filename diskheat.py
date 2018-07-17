import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time


"""
Here is an example of solving heat diffusion equation on a uniform disk in radial coordinates.
dT/dt = alpha*del^2(T) + heating/(rho*Cp) (https://en.wikipedia.org/wiki/Heat_equation)
Temperature is fixed at the edge. Initial tempearture is uniform.
Heat is generated within a disk with gaussian distribution.
"""

# input parameters
#----------------------------------------------------------------------------------------------------------------------------
rho = 4.826              # density, g/cm^3
Cp = 0.3369              # specific heat capacity,  J/(g*K)
alpha = 0.123            # thermal diffusivity, cm^2/s

R = 3*10**(-4)           # radius of the disk, cm
nr = 512                 # number of the radial slices

dr = R/nr                # delta r
dr2 = dr**2
rv = np.arange(0.,R, dr) # radius values

dt =  dr**2./(4*alpha)   # time step, s. in order for solution to be stable, must be smaller than dr**2./(2*alpha) 

Tcool = 300              # initial temperature, K. And boundary condition at the edge, e.g. keep the edge at Tcool tempearature 
u0 = np.ones(nr)*Tcool  # initial temperature distribution
power = 100*10**(-3)     # total power of the heat source, W
sigmahs = 5.*10**(-5)    # standard deviation of the heat source distribution, cm

# heat source distribution
#----------------------------------------------------------------------------------------------------------------------------
heating = power*(1./(sigmahs**2*2*np.pi))*np.exp(-(rv**2)/(2*sigmahs**2))
heatsource = heating/(rho*Cp)

# solution
#----------------------------------------------------------------------------------------------------------------------------
D0 = 2*alpha*dt/(dr2)

D1 = alpha*dt*(rv[1:-1] + dr/2.)/(dr2*rv[1:-1])
D2 = 1. - 2*alpha*dt/(dr2)
D3 = alpha*dt*(rv[1:-1] - dr/2.)/(dr2*rv[1:-1])

Heatscld  = heatsource[1:-1]*dt
Heatscld0 = heatsource[0]*dt


@jit # comment this line for no jit option 
def timestep(u):  # calculates next in time temperature distribution based on input temperature dstribution u
    u[0] = u[0] + D0*(u[1] - u[0]) + Heatscld0              # boundary condition at the center
    u[1:-1] = D1*u[2:] + D2*u[1:-1]  + D3*u[:-2] + Heatscld
    u[-1]  = Tcool                                          # boundary conditions at the edge

    return u.copy()


# main loop
#----------------------------------------------------------------------------------------------------------------------------
Ndt = 10**6          # number of time steps to calculate the solution for
Ns  = 100            # store the data once in Ns*dt time 
Ndata = int(Ndt/Ns)  # number of the data points

m=0
tol = 1
temp_center = np.zeros(Ndata)
u = u0

stime = time.time()  # strat time of the main loop, for performance measurement
for i in range(Ndt):
    u = timestep(u)
    if i%Ns==0:
        temp_center[i/Ns] = u[0]

timetaken = time.time() - stime


print "Calculation finished ... " 
print "time taken = %1.3f"%timetaken, ' seconds'
print "%1.4e seconds per iteration"%(timetaken/Ndt)
print "Last simulation time, ", Ndt*dt 

# plot the results
#----------------------------------------------------------------------------------------------------------------------------
times = np.arange(Ndata)*dt*Ns

plt.plot(times, temp_center)
plt.title("Temperature at the center of the disk")

plt.figure()
plt.plot(rv, u)
plt.title("Final temperature dstribution ")
plt.show()