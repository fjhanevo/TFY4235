import numpy as np
import matplotlib.pyplot as plt

"""
TFY4235 Computational Phyiscs Exam 2024.
Problem 3.
"""

# Parameters
a = 0   
b = 10
dx = 0.1
dt = 0.01
T = 2
x0 = 5
sigma = 0.1

Nx = int((b-a) / dx) + 11
Nt = int(T/dt) + 1

# Spatial grid
X = np.linspace(a,b,Nx)

# Initial condition 
V0 = np.exp((-(X-x0)**2)/(2*sigma**2))

def euler_explicit(V,dx,dt,Nx,Nt):
    """ Implementation of the Euler Explicit method for the cable equation """
    # Create matrix to hold all time steps
    V_time = np.zeros((Nx,Nt))
    V_time[:,0] = V

    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            V_time[i, n+1] = V_time[i,n] + dt * (
                    (V_time[i+1,n] - 2*V_time[i,n] + V_time[i-1,n]) 
                    / dx**2 - V_time[i,n])
        
        # Neumann boundary conditions: dV/dx = 0 at x = a,b
        V_time[0,n+1] = V_time[1,n+1]    # l.h.s
        V_time[-1, n+1] = V_time[-2,n+1] # r.h.s

    return V_time


V_time_explicit = euler_explicit(V0,dx,dt,Nx,Nt)
plt.figure(figsize=(10,6))
time_steps_to_plot = [0,int(Nt/4),int(Nt/2),int(3*Nt/4),Nt-1]
for i in time_steps_to_plot:
    plt.plot(X,V_time_explicit[:,i],label=f'{i*dt:.2f}s')

plt.tight_layout()
plt.show()




