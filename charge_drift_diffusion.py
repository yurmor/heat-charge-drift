import numpy as np
import time 
from scipy.constants import e, epsilon_0

"""
here solving pair of charge drift-diffusion equations
for elctrons and holes in a semiconductor, plus Poisson's equation to calculate electric field

dn/dt = G - R + Dce*d^2(n)/dx^2 + 1/e*d(-e*mue*E)/dx
dp/dt = G - R + Dch*d^2(p)/dx^2 + 1/e*d(+e*muh*E)/dx
d^2(phi)/dx^2 = -rho/(epsilon*epsilon0)

working in radial coordinates, over a disk with radius R
chrage distribution along z-direction is uniform 

n is the chrage density (1/cm^3) 
R = An + Bnp + 0.5*C(n*p^2 + p*n^2), recombination (1/(s*cm^3))
E - electric field (V/cm)

A - nonradiative recombiniation rate, 
B - bimolecular radiative recombination rate,
C - nonradiative Auger recombinatino rate

G - generation (1/(s*cm^3)) 

"""

# input parameters
#----------------------------------------------------------------------------------------------------------------------------
abspower = 10**(-3)          # total absorbed power, W
Eg = 1.417                   # band gap, eV
absrate =  abspower/(Eg*e)   # absorption rate, 1/s. Total number of absorbed photons per second
sigmal = 5.*10**(-4)         # standard deviation of the laser spot distribution, cm

A = 2.72*10**(5)             # monomolecular non-radiative recombination rate constant, 1/s
B = 1.7*10**(-10)            # bimocular recombination rate constant, cm^3/s
C = 7*10**(-30)              # Auger recombination rate constant, cm^6/s

Dce = 10.               	 # electron diffusivity cm^2/s

T = 10**(-5)                 # thickness of the disk, cm
R = 40*10**(-4)              # radius of the disk, cm
nr = 256                     # number of the radial slices

dr = R/nr                    # delta r
dr2 = dr**2
rv = np.arange(nr)*dr        # radius values

dt = 1.*dr2/(4*Dce)          # time step, s

# initial conditions
#----------------------------------------------------------------------------------------------------------------------------
n0 = np.zeros(nr)        # initial electron charge density distribution, 1/cm^3
p0 = np.zeros(nr)        # initial hole charge density distribution, 1/cm^3
phi0 = np.zeros(nr)      # initial potential distribution, V

# generation rate density distribution, 1/()
G_density = absrate*(1./(sigmal**2*(2.*np.pi)*T))*np.exp(-rv**2/(2*sigmal**2))


# calculate some constants that we will use frequently to save calculation time
#----------------------------------------------------------------------------------------------------------------------------
rvin = rv[1:-1]  # radius values inside, without boundary 

Gscld = G_density*T
Ascld = A
Bscld = B/T
Cscld = 0.5*C/T**2

De_plus = Dce*(rvin + dr/2)/(rvin*dr2)
De_minus = Dce*(rvin - dr/2)/(rvin*dr2)

# solution functions
#----------------------------------------------------------------------------------------------------------------------------



def timestep(nl, dt):
	n = np.zeros(len(nl))

	nin = nl[1:-1]
	
	GRe_rate = Gscld - Ascld*nl - Bscld*nl*nl - Cscld*(2*nl**3)  # Generation rate for electrons
	
	ndd_rate  = nl[2:]*De_plus + nl[:-2]*De_minus + nin*( -2*Dce/dr2 )
	
	n[1:-1] = nin + ndd_rate*dt + GRe_rate[1:-1]*dt

	#---------------------------------------------------------------------
	
	n[0] = nl[0] + 2*(Dce/dr2)*(nl[1]-nl[0])*dt + GRe_rate[0]*dt 

	
	#Boundary condition for charge density
	#n[-1] = n[-2]
	
	n[-1] = nl[-1] + GRe_rate[-1]*dt  - Dce*dt/dr2*(nl[-1] - nl[-2])*(nr-1.5)/(nr-1.)
	
	return n

# main loop
#----------------------------------------------------------------------------------------------------------------------------
Ndt = 100000           # number of time steps to calculate the solution for
Ns  = 100               # store the data once in Ns*dt time 
Ndata = int(Ndt/Ns)     # number of the data points

ntransient = np.zeros(Ndata) # store time dependent electron concetration here

stime = time.time()     # strat time of the main loop, for performance measurement

n = n0

for i in range(Ndt):
	n = timestep(n, dt)
	if i%Ns==0:
		ntransient[i/Ns] = n[0]

timetaken = time.time() - stime

# convert densities back to 1/cm^3
n,  ntransient = n/T, ntransient/T

print "Calculation finished ... " 
print "time taken = %1.3f"%timetaken, ' seconds'
print "%1.4e seconds per iteration"%(timetaken/Ndt)
print "Last simulation time, ", Ndt*dt 
print "Last electron concentration at the center = ", n[0], " 1/cm^3"


# plot the results
#----------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
times = np.arange(Ndata)*dt*Ns

plt.figure()
plt.plot(times*10**(6), ntransient)

plt.figure()
plt.plot(rv, n)

plt.show()