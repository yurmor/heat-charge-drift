import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time

"""
--isolated boundary conditions
--check r=0 

"""

"""
Here is an example of solving heat diffusion equation on a uniform cylinder in radial coordinates.
dT/dt = alpha*del^2(T) + heating/(rho*Cp) (https://en.wikipedia.org/wiki/Heat_equation)

---Temperature is fixed at the edge. Initial tempearture is uniform.
---Heat is generated within a disk with gaussian distribution.

Sometimes can be faster to use @jit from Numba, experiment with it
"""

# input parameters
#----------------------------------------------------------------------------------------------------------------------------
rho = 4.826              # density, g/cm^3
Cp = 0.3369              # specific heat capacity,  J/(g*K)
alpha = 0.123            # thermal diffusivity, cm^2/s

R = 5*10**(-4)           # radius of the disk, cm
h = 1.*10**(-4)          # height of the cylinder, cm
nr = 512                 # number of the radial slices
nz = 100                 # number of the z-direction slices

dr = R/nr                # delta r
dr2 = dr**2

dz = h/nz                # delta z
dz2 = dz**2

rv = np.arange(nr)*dr # radius values
zv = np.arange(nz)*dz # z values

dt =  dr**2*dz**2/(4*alpha*(dr**2 + dz**2))   # time step, s. in order for solution to be stable, must be smaller than dr**2*dz**2/(2*alpha*(dr**2 + dz**2)) 

T0 = 300                 # initial temperature, K.
Tside = 300              # boundary condition at the side of the cylinder
Tbottom = 290            # boundary condition at the bottom surface of the cylinder
Ttop = 310               # boundary condition at the top surface of the cylinder

u0 = np.ones((nz, nr))*T0  # initial temperature distribution

power = 100*10**(-3)     # total power of the heat source, W
sigmahs = 5.*10**(-5)    # standard deviation of the heat source distribution, cm

# heat source distribution
#----------------------------------------------------------------------------------------------------------------------------
heating = power*(1./h)*(1./(sigmahs**2*2*np.pi))*np.exp(-(rv**2)/(2*sigmahs**2))
heatsource = heating/(rho*Cp)

# solution
#----------------------------------------------------------------------------------------------------------------------------
Dc0 = 2*alpha*dt/(dr2)

Dc1 = alpha*dt*(rv[1:-1] + dr/2.)/(dr2*rv[1:-1])
Dc2 = 1. - 2*alpha*dt/(dr2)
Dc3 = alpha*dt*(rv[1:-1] - dr/2.)/(dr2*rv[1:-1])

Dcz = alpha*dt/(dz2)

Heatscld  = dz*heatsource[1:-1]*dt
Heatscld0 = dz*heatsource[0]*dt


#@jit # comment this line for no jit option 
def timestep(u):  # calculates next in time temperature distribution based on input temperature dstribution u

    u[:,0] = u[:,0] + Dc0*(u[:,1] - u[:,0]) + Heatscld0              # boundary condition at the center
    u[1:-1, 1:-1] = Dc1*u[1:-1, 2:] + Dc2*u[1:-1, 1:-1]  + Dc3*u[1:-1, :-2] + Dcz*(u[2:, 1:-1] -2.*u[1:-1, 1:-1] + u[:-2, 1:-1]) + Heatscld
    u[:,-1]  = Tside                                         # boundary conditions at the edge
    u[0,:] = Tbottom
    u[-1,:] = Ttop
    return u.copy()


# main loop
#----------------------------------------------------------------------------------------------------------------------------
Ndt = 10**4        # number of time steps to calculate the solution for
Ns  = 100            # store the data once in Ns*dt time 
Ndata = int(Ndt/Ns)  # number of the time steps to store the data for

m=0
tol = 1
temp_center = np.zeros(Ndata)
u = u0

stime = time.time()
for i in range(Ndt):
    u = timestep(u)
    if i%Ns==0:
        temp_center[i/Ns] = u[nz/2, 0]

timetaken = time.time() - stime

print u[1,0], u[-2,0]

print "Calculation finished ... " 
print "time taken = %1.3f"%timetaken, ' seconds'
print "%1.4e seconds per iteration"%(timetaken/Ndt)
print "Last simulation time, ", Ndt*dt 

# plot the results
#----------------------------------------------------------------------------------------------------------------------------
times = np.arange(Ndata)*dt*Ns

plt.plot(times, temp_center)
plt.title("Temperature at the center of the cylinder")

plt.figure()
plt.plot(rv, u[1,:])
plt.plot(rv, u[nz/2,:])
plt.plot(rv, u[-2,:])
plt.title("Final temperature dstribution ")


plt.figure()
plt.plot(zv, u[:,0])
plt.show()