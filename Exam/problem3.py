import numpy as np
import matplotlib.pyplot as plt

"""
TFY4235 Computational Phyiscs Exam 2024.
Problem 3.
"""

# Parameters
Nx = 100 
Nt = 1000
a,b = 0.,1.
x0 = 0.5
sigma = 0.1

dx = (b-a)/Nx
dt = dx**2 / 2

# Spatial grid
X = np.linspace(a,b,Nx)
T = np.linspace(a,b,Nt)

# Initial condition 
V0 = np.exp((-(X-x0)**2)/(2*sigma**2))

def euler_explicit(V,dx,dt,Nx,Nt):
    """ Implementation of the Euler Explicit method for the cable equation """
    # Create matrix to hold all time steps
    V_time = np.zeros((Nx,Nt))
    V_time[:,0] = V

    for n in range(0, Nt-1):
        V_time[1:-1, n+1] = V_time[1:-1,n] + dt * (
                (V_time[2:,n] - 2*V_time[1:-1,n] + V_time[:-2,n]) 
                / dx**2 - V_time[1:-1,n])
    
        # Neumann boundary conditions: dV/dx = 0 at x = a,b
        V_time[0,n+1] = V_time[1,n+1]    # l.h.s
        V_time[-1, n+1] = V_time[-2,n+1] # r.h.s

    return V_time


V_time_explicit = euler_explicit(V0,dx,dt,Nx,Nt)
plt.figure(figsize=(10,6))
time_steps_to_plot = [0,int(Nt/4),int(Nt/2),int(3*Nt/4),Nt-1]
for i in time_steps_to_plot:
    plt.plot(X,V_time_explicit[:,i],label=f't={i*dt:.2f}s')

plt.title('Explicit Euler')
plt.xlabel('Position [m]?')
plt.ylabel('Voltage?')
plt.legend()
plt.tight_layout()
plt.show()




