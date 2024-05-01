import matplotlib.pyplot as plt
import numpy as np

""" 
TFY4235 Computational Physics Exam 2024
Problem 3.7
"""

# Parameters
Nx = 100 
Nt = 1000
a,b = 0.,1.
t0,tf = 0., 1.
x0 = (b-a)/2 
sigma =0.1 

# Constants from task 3.7a) 
l = 0.18        # [mm]
tau = 2.0       # [ms]
gamma = 0.5     # [mV^-1]
V_star = -40    # [mV]  
V_Na = 56      # [mV] Nernst 
V_K = -76       # [mV] Nernst
V_mem = -70     # [mV]

V_appl = -50    # [mV] change this for other tasks
g_K = 5

dx = (b-a)/Nx
# dt = dx**2 / 2
dt = 1/Nt

# Spatial grid
X = np.linspace(a,b,Nx)

# Time grid
T = np.linspace(t0,tf,Nt)

# Initial condition 
V0 = (V_appl-V_mem)*np.exp(-((X-x0)**2)/(2*l**2)) + V_mem


def gNa(V,gamma,V_star):
    """ Computes g_Na(V) """
    return (100/(1+np.exp(gamma*(V_star-V))) + 1/5)

def explicit_euler(V0,dx,dt,Nx,Nt):
    V_time = np.zeros((Nx, Nt))
    
    # Apply initial condition
    V_time[:, 0] = V0
    
    # Vectorize the operations to avoid double for-loop
    for n in range(0, Nt - 1):
        gNa_value = gNa(V_time[:, n], gamma, V_star)
        d2V_dx2 = (V_time[2:, n] - 2 * V_time[1:-1, n] + V_time[:-2, n]) / dx**2
        term1 = l**2 * d2V_dx2
        term2 = (gNa_value[1:-1] / g_K) * ((V_time[1:-1, n] - V_Na) / gNa_value[1:-1] + (V_time[1:-1, n] - V_K) / g_K)
        V_time[1:-1, n + 1] = V_time[1:-1, n] + dt * (term1 - term2 / tau)

        # Neumann boundary conditions: dV/dx = 0 at x = a, b
        V_time[0, n + 1] = V_time[1, n + 1]  # l.h.s
        V_time[-1, n + 1] = V_time[-2, n + 1]  # r.h.s

    return V_time

def plot_method(V,X,time_steps,title,filename=None):
    """ Plots the time evolution of a single method"""

    plt.figure(figsize=(10,6))
    for t in time_steps:
        index = int(t*(Nt-1))
        plt.plot(X,V[:,index],label=f'{t=:.2f}s')
    plt.title(title)
    plt.xlabel('Position [m]')
    plt.ylabel('Voltage [V]')
    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.tight_layout()
        plt.show()


time_steps = [T[1],T[25],T[50],T[450],T[900]]
V_explicit = explicit_euler(V0,dx,dt,Nx,Nt)

plot_method(V_explicit,X,time_steps,'test')
