import numpy as np
import matplotlib.pyplot as plt
"""
TFY4235 Computational Phyiscs Exam 2024.
Problem 3.5
"""

# Parameters
Nx = 100        # Spacial interval
Nt = 1000       # Temporal interval 
t0,tf = 0., 1.  # Temporal domain 
a,b = 0.,10.    # Spacial domain
x0 = (b-a)/2    # Midpoint
sigma = 0.1     # Width of the gaussian impulse
l = 1.0         # lambda value from task
tau = 1.0       # tau value from task

dx = (b-a)/Nx   # Spatial step size
dt = (tf-t0)/Nt # Temporal step size

X = np.linspace(a,b,Nx)     # Spatial grid
T = np.linspace(t0,tf,Nt)   # Temporal grid

def initial_condition(x):
    """ Implementation of a narrow Gaussian beam as initial condition """
    return  2.5/(np.sqrt(2*np.pi))*np.exp(-((x-x0)/sigma)**2)

def explicit_euler(V0,dx,dt,Nx,Nt):
    """ Implementation of the Euler Explicit scheme for the cable equation """
    # Create matrix to store results 
    V_time = np.zeros((Nx,Nt))
    
    # Apply initial condition
    V_time[:,0] = V0
    
    # Vectorize the operations to avoid double for-loop
    for n in range(0, Nt-1):
        V_time[1:-1, n+1] = V_time[1:-1,n] + dt * (
                (V_time[2:,n] - 2*V_time[1:-1,n] + V_time[:-2,n]) 
                / dx**2 - V_time[1:-1,n])
    
        # Neumann boundary conditions: dV/dx = 0 at x = a,b
        V_time[0,n+1] = V_time[1,n+1]    # l.h.s
        V_time[-1, n+1] = V_time[-2,n+1] # r.h.s

    return V_time

def implicit_euler(V0,dx,dt,Nx,Nt):
    """ Implementation of the implicit Euler scheme for the cable equation """
    # Import solve_banded to avoid having to invert a tridiagonal matrix
    from scipy.linalg import solve_banded
    
    # CFL number
    alpha = dt/(dx**2)

    # Create matrix to store results
    V_time = np.zeros((Nx,Nt))

    # Apply initial condition
    V_time[:,0] = V0

    
    lower = np.ones(Nx-1) * (-alpha)
    upper = np.ones(Nx-1) * (-alpha)
    diagonal = np.ones(Nx) * (1 + 2 * alpha)

    # Neumann BCs with no flux at the boundaries:
    diagonal[0] = 1 + alpha
    diagonal[-1] = 1 + alpha

    # Store the matrix in a banded format for solve_banded
    A_banded = np.vstack((np.append(0, upper), diagonal, np.append(lower, 0)))

    # Time stepping
    for n in range(1, Nt):
        V_time[:, n] = solve_banded((1, 1), A_banded, V_time[:, n-1])

    return V_time

def crank_nicolson(V0,dx,dt,Nx,Nt):
    """ Implementation of the Crank-Nicolson scheme, combining implicit and explicit schemes """

    V_explicit = explicit_euler(V0,dx,dt,Nx,Nt)
    V_implicit = implicit_euler(V0,dx,dt,Nx,Nt)

    return 0.5* (V_explicit + V_implicit)

def analytical_solution(V0,x,t,l,tau,x0):
    """ Implementation of the unbounded analytical solution """

    exponent = -((x-x0)**2)/(4*(l**2/tau)*t) -t/tau
    return (V0/np.sqrt(4*np.pi*(l**2/tau)*t))*np.exp(exponent)

def plot_method(V,X,time_steps,title,filename=None):
    """ Plots the time evolution of a single method and compares it with the analytical solution """

    plt.figure(figsize=(10,6))
    for t in time_steps:
        # index = int(t*(Nt-1))
        index = int(t / (T[-1] / Nt))
        V_Analytical = analytical_solution(V_t,X,t,l,tau,x0)
        plt.plot(X,V[:,index],label=f'{title}: t={T[index]:.2f}s', linestyle='dashed')
        plt.plot(X,V_Analytical,label=f'Analytical: t={t:.2f}s')
    plt.xlabel(f'x [m]',fontsize=18)
    plt.ylabel(f'V(x,t) [V]',fontsize=18)
    plt.legend(fontsize=15)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.tight_layout()
        plt.show()


 
V0 = initial_condition(X)
V_explicit = explicit_euler(V0,dx,dt,Nx,Nt)
V_implicit = implicit_euler(V0,dx,dt,Nx,Nt) 
V_crank_nicolson = crank_nicolson(V0,dx,dt,Nx,Nt)
V_t = 0.165 
time_steps = [T[10], T[20], T[30], T[40], T[80],T[100]]
plot_method(V_explicit,X,time_steps,'Explicit Euler','Figures/3_5_explicit.png')
plot_method(V_implicit,X,time_steps,'Implicit Euler','Figures/3_5_implicit.png')
plot_method(V_crank_nicolson,X,time_steps,'Crank-Nicolson','Figures/3_5_cn.png')

