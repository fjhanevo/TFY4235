import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
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

def explicit_euler(V0,dx,dt,Nx,Nt):
    """ Implementation of the Euler Explicit scheme for the cable equation """
    # Create matrix to hold all time steps
    V_time = np.zeros((Nx,Nt))
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
    """ Implicit Euler scheme for the cable equation """
    
    alpha = 1*dt/(dx**2) # CFL number, D = 1
    # Setup matrix for spatial discretization with Neumann bc's
    diag = np.ones(Nx) * (1+2*alpha)
    off_diag = np.ones(Nx-1)*(-alpha)
    A = diags([diag,off_diag,off_diag], [0,-1,1], format='csr')
    A[0,1] = -2*alpha
    A[-1,-2] = -2*alpha

    # Identity matrix for the l.h.s of the scheme
    I = identity(Nx,format='csr')
    V_time = np.zeros((Nx,Nt))
    V_time[:,0] = V0

    # Time stepping
    for i in range(1,Nt):
        b = V0 + dt * V0
        V0 = spsolve(I+dt*A,b)
        V_time[:,i] = V0
    return V_time

def crank_nicolson(V0,dx,dt,Nx,Nt):
    """ Crank-Nicolson scheme for the cable equation """
    alpha = 1*dt/(dx**2)    # CFL number, D = 1

    # Setup matrix for spatial discretization with Neumann bc's
    diag = np.ones(Nx) * (1+2*alpha)
    off_diag = np.ones(Nx-1) * (-alpha)
    A =  diags([diag,off_diag,off_diag], [0,-1,1],format='csr')
    A[0,1] = -2 * alpha
    A[-1,-2] = -2*alpha

    # Matrices for CN-scheme
    I = identity(Nx,format='csr')
    CN_lhs = I-0.5*dt*A-0.5*dt*I
    CN_rhs = I+0.5*dt*A+0.5*dt*I

    # Matrix to store results, set inital condition
    V_time = np.zeros((Nx,Nt))
    V_time[:,0] = V0

    # Time stepping
    for i in range(1,Nt):
        b = CN_rhs @ V0
        V0 = spsolve(CN_lhs,b)
        V_time[:,i] = V0
    
    return V_time

def analytical_solution(V0,x,t,x0):
    """ Implementation of the analytical solution """
    # lamda = tau = 1.0
    return (V0/np.sqrt(4*np.pi*t))*np.exp(-((x-x0)**2)/4*t-t)

def test_analytical_solution(V0,x,t,x0):
    pref = np.sqrt(4*np.pi*t)
    V_analytical = np.zeros((len(x),len(t)))
    t = t.reshape(1,-1)
    for idx,i in enumerate(t):
        # if i == 0:
        #     V_analytical[:,idx] = 0
        # else:
        exp_part = np.exp(-((x-x0)**2).reshape(-1,1)/(4*t)-t)
        V_analytical[:,idx] = V0/pref[idx]*exp_part.squeeze()
    return V_analytical


def plot_methods(X,V_explicit,V_implicit,V_cn,time_steps):
    """ Plots the method in a 2x2 square for easy comparison """
    fig, axs = plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)
    
    # Flatten to make indexing easier
    axs = axs.flatten()
    # Plot information
    # titles = ['Explicit Euler','Implicit Euler','Crank-Nicolson', 'Analytical Solution']
    titles = ['Explicit Euler','Implicit Euler','Crank-Nicolson']

    # v_matrices = [V_explicit,V_implicit,V_cn,V_analytical]
    v_matrices = [V_explicit,V_implicit,V_cn]

    # Plot each method in its subplot
    for i, V in enumerate(v_matrices):
        ax = axs[i]
        for t in time_steps:
            index = int(t*(Nt-1))
            # ax.plot(X,V[:,index],label=f't={t*dt:.2f}s')
            ax.plot(X,V[:,index],label=f't={t:.2f}s')
        ax.set_title(titles[i])
        ax.set_xlabel('Position [m]')
        ax.set_ylabel('Voltage [V]')
        ax.legend()

    plt.tight_layout()
    plt.show()


V_explicit = explicit_euler(V0,dx,dt,Nx,Nt)
V_implicit = implicit_euler(V0,dx,dt,Nx,Nt) 
V_crank_nicolson = crank_nicolson(V0,dx,dt,Nx,Nt)
# V_analytical = test_analytical_solution(V0,X,T,x0)
# time_steps = [0,int(Nt/4),int(Nt/2),int(3*Nt/4),Nt-1]
time_steps = [T[1],T[25],T[800]]
plot_methods(X,V_explicit,V_implicit,V_crank_nicolson,time_steps)

# def plot_test(timestep):
#     for t in timestep:
#         # V = test_analytical_solution(V0,X,t,x0)
#         plt.plot(X,V_analytical[:,t],label=f't={t}')
#     plt.legend()
#     plt.show()
#
# plot_test(time_steps)
