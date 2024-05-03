import matplotlib.pyplot as plt
import numpy as np

""" 
TFY4235 Computational Physics Exam 2024
Problem 3.7
"""

# Parameters
Nx = 100 
Nt = 1000
a,b = 0.,2.
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
dt = 0.0001

# Spatial grid
X = np.linspace(a,b,Nx)

# Time grid
T = np.linspace(t0,tf,Nt)

# Initial condition 
V0 = (V_appl-V_mem)*np.exp(-((X-x0)**2)/(2*l**2)) + V_mem

sodium_channel_index = int((0.25-x0)/dx)

def gNa(V, sodium_channel_idx=None):
    """ Computes g_Na(V) with increased conductance at a specific index. """
    gNa_values = np.full_like(V, 100 / (1 + np.exp(gamma * (V_star - V))) + 1 / 5)
    if sodium_channel_idx is not None:
        gNa_values[sodium_channel_idx] = 1000 / (1 + np.exp(gamma * (V_star - V[sodium_channel_idx]))) + 1 / 5
    return gNa_values

def explicit_euler(V0,dx,dt,Nx,Nt):
    """ Implementation of the Explicit Euler scheme to solve eq.15 """

    global l,tau,g_K, V_K

    # Store results
    V_time = np.zeros((Nx, Nt))
    
    # Apply initial condition
    V_time[:, 0] = V0
    
    # Vectorize the operations to avoid double for-loop
    for n in range(0, Nt - 1):
        # Compute the voltage-dependent sodium channel permeability for current time step
        # gNa_value = gNa(V_time[:, n])
        gNa_value = gNa(V_time[:,n],sodium_channel_index)

        # Compute the second spatial derivative using finite differences
        d2V_dx2 = (V_time[2:, n] - 2 * V_time[1:-1, n] + V_time[:-2, n]) / dx**2

        # Calculate the term related to diffusion effect
        term1 = l**2 * d2V_dx2

        # Calculate term involving channel permeabilities and Nernst potentials
        term2 = (gNa_value[1:-1] / g_K) * ((V_time[1:-1, n] - V_Na) + (V_time[1:-1, n] - V_K))
        # term2 = gNa_value[1:-1]  * ((V_time[1:-1, n] - V_Na)/gNa_value[1:-1] + (V_time[1:-1, n] - V_K)/g_K)

        # Update vals for the next time step using explicit Euler method
        V_time[1:-1, n + 1] = V_time[1:-1, n] + (dt/tau)* (term1 - term2)
        # V_time[1:-1, n + 1] = V_time[1:-1, n] + dt* (term1 - term2/tau)

        # Apply Neumann bc's 
        V_time[0, n + 1] = V_time[1, n + 1]  # l.h.s
        V_time[-1, n + 1] = V_time[-2, n + 1]  # r.h.s

    return V_time

def find_min_impulse(Nx,Nt,a,b,dx,dt,threshold,V_mem):
    min_impulse=None
    for V_appl in np.arange(-60,-30,1):
        V0 = (V_appl - V_mem) * np.exp(-((np.linspace(a,b,Nx)-x0)**2)/(2*l**2))+V_mem
        V_time = explicit_euler(V0,dx,dt,Nx,Nt)

        if np.any(V_time[sodium_channel_index] >= threshold):
            min_impulse = V_appl
            break
    return min_impulse

def plot_method(V,X,time_steps,filename=None):
    """ Plots the time evolution of a single method"""

    plt.figure(figsize=(10,6))
    for t in time_steps:
        index = int(t*(Nt-1))
        plt.plot(X,V[:,index],label=f'{t=:.2f}s')
    plt.xlabel('x [m]',fontsize=18)
    plt.ylabel('V(x,t) [V]',fontsize=18)
    plt.legend(fontsize=15)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.tight_layout()
        plt.show()


# time_steps = [T[1],T[25],T[50],T[450],T[900]]
time_steps = [0.0,0.2,0.4,0.6,0.8,1.]
V_explicit = explicit_euler(V0,dx,dt,Nx,Nt)
print(dx)
print(dt)
threshold = -65
min_impulse = find_min_impulse(Nx, Nt, a, b, dx, dt, threshold, V_mem)
print(f"Minimum impulse strength to induce an action potential: {min_impulse} mV")
# plot_method(V_explicit,X,time_steps,'Figures/3_7_Vappl_30')
plot_method(V_explicit,X,time_steps)

