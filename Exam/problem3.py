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

    
# V_time_explicit = explicit_euler(V0,dx,dt,Nx,Nt)
# V_time_implicit = implicit_euler(V0,dx,dt,Nx,Nt) 
V_crank_nicolson = crank_nicolson(V0,dx,dt,Nx,Nt)
plt.figure(figsize=(10,6))
time_steps_to_plot = [0,int(Nt/4),int(Nt/2),int(3*Nt/4),Nt-1]
for i in time_steps_to_plot:
    plt.plot(X,V_crank_nicolson[:,i],label=f't={i*dt:.2f}s')

plt.title('Explicit Euler')
plt.xlabel('Position [m]?')
plt.ylabel('Voltage?')
plt.legend()
plt.tight_layout()
plt.show()



