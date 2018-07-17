import numpy as np
import time 
import matplotlib.pyplot as plt

"""
here solving dn/dt = G-R 
where n is charge density (1/cm^3) 
R = An + Bn^2 + Cn^3, recombination (1/(s*cm^3))

A - nonradiative recombiniation rate, 
B - bimolecular radiative recombination rate,
C - nonradiative Auger recombinatino rate

G - generation (1/(s*cm^3)) 
electrons and holes concetrtions assumed to be the same
"""

# input parameters
#----------------------------------------------------------------------------------------------------------------------------
n0 = 0                  # initial charge density, 1/cm^3
G = 10.**25             # generation rate density, 1/(s*cm^3)

A = 2.72*10**(5)        # monomolecular non-radiative recombination rate constant, 1/s
B = 1.7*10**(-10)       # bimocular recombination rate constant, cm^3/s
C = 7*10**(-30)         # Auger recombination rate constant, cm^6/s

dt = 10**-11            # time step, s


# solution function
#----------------------------------------------------------------------------------------------------------------------------
def timestep(nlast):
    n = nlast + dt*(G - (A*nlast + B*nlast**2 + C*nlast**3))
    return n

# main loop
#----------------------------------------------------------------------------------------------------------------------------
Ndt = 10**6             # number of time steps to calculate the solution for
Ns  = 100               # store the data once in Ns*dt time 
Ndata = int(Ndt/Ns)     # number of the data points

ntransient = np.zeros(Ndata) # store time dependent electron concetration here


stime = time.time()     # strat time of the main loop, for performance measurement
nlast = n0
for i in range(Ndt):
    nlast = timestep(nlast)
    if i%Ns==0:
        ntransient[i/Ns] = nlast

timetaken = time.time() - stime

print "Calculation finished ... " 
print "time taken = %1.3f"%timetaken, ' seconds'
print "%1.4e seconds per iteration"%(timetaken/Ndt)
print "Last simulation time, ", Ndt*dt 
print "Last electron concentratino = ", nlast, " 1/cm^3"


# plot the results
#----------------------------------------------------------------------------------------------------------------------------
times = np.arange(Ndata)*dt*Ns

plt.plot(times*10**(6), ntransient)
plt.xlabel('Times (us)')
plt.ylabel('Electron concentration (1/cm^3)')
plt.show()