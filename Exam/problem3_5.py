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
a,b = 0.,10.
t0,tf = 0., 1.
# a,b = 0.,10.
x0 = (b-a)/2 
sigma =0.1 

dx = (b-a)/Nx
# dt = dx**2 / 2
dt = 1/Nt
# Spatial grid
X = np.linspace(a,b,Nx)
T = np.linspace(t0,tf,Nt)

# Initial condition 
V0 = np.exp((-(X-x0)**2)/(2*sigma**2))

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
    alpha = dt/(2*dx**2)

    # Create matrix to store results
    V_time = np.zeros((Nx,Nt))

    # Apply initial condition
    V_time[:,0] = V0

    
    lower = np.ones(Nx-1) * (-alpha)
    upper = np.ones(Nx-1) * (-alpha)
    diagonal = np.ones(Nx) * (1 + 2 * alpha)

    # Adjust boundary conditions for A if needed
    # For Neumann BCs with no flux at the boundaries:
    diagonal[0] = 1 + alpha
    diagonal[-1] = 1 + alpha

    # Store the matrix in a banded format for `solve_banded`
    A_banded = np.vstack((np.append(0, upper), diagonal, np.append(lower, 0)))

    # Time stepping
    for n in range(1, Nt):
        V_time[:, n] = solve_banded((1, 1), A_banded, V_time[:, n-1])

    return V_time

def get_cn_matrices(Nx,dx,dt):
    """
    Initializes the A and B matrix used for the Crank-Nicolson scheme. 
    Applies Neumann boundary conditions
    """
    diag = np.ones((Nx))
    alpha = dt/(2*dx**2)

    A = (-alpha/2) * np.diag(diag[1:], -1) +\
        (-alpha/2) * np.diag(diag[1:],1) +\
        (1+alpha)*np.diag(diag)

    B = (alpha/2) * np.diag(diag[1:], -1) +\
        (alpha/2) * np.diag(diag[1:],1) +\
        (1-alpha) * np.diag(diag)
   
    # Apply Neumann bc's
    A[0,0] = 1 + alpha
    A[-1,-1] = 1 + alpha
    B[0,0] = 1 - alpha
    B[-1,-1] = 1 - alpha

    return A,B

def crank_nicolson(V0,dx,dt,Nx,Nt):
    """ Implementation of the Crank-Nicolson scheme for the cable equation """
 
    # Initialize V_time to store results 
    V_time = np.zeros((Nx,Nt))

    # Apply initial condition
    V_time[:,0] = V0

    # Get matrices
    A, B = get_cn_matrices(Nx,dx,dt)

    # Solve the equation
    for i in range(Nt-1):
        u = B@V_time[:,i]
        V_time[:,i+1] = np.linalg.solve(A,u)
    
    return V_time 

def analytical_solution(V0,x,t,x0):
    """ Implementation of the analytical solution """
    # lamda = tau = 1.0
    return (V0/np.sqrt(4*np.pi*t))*(np.exp(-((x-x0)**2)/(4*t-t)))

def plot_method(V,X,time_steps,title,filename=None):
    """ Plots the time evolution of a single method"""

    plt.figure(figsize=(10,6))
    for t in time_steps:
        index = int(t*(Nt-1))
        plt.plot(X,V[:,index],label=f't={t=:.2f}s')
    plt.title(title)
    plt.xlabel('Position [m]')
    plt.ylabel('Voltage [V]')
    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.tight_layout()
        plt.show()

def plot_analytical(X,time_steps, filename=None):
    """ Plots only the analytical solution """
    for t in time_steps:
        V = analytical_solution(V_t,X,t,x0)
        plt.plot(X,V,label=f't={t:.2f}s')
    plt.title('Analytical solution')
    plt.xlabel('Position [m]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.tight_layout()
        plt.show()



""" Keeping this here for comparison"""
def plot_methods(X,V_explicit,V_implicit,V_cn,time_steps):
    """ Plots the method in a 2x2 square for easy comparison """
    # fig, axs = plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)
    fig, axs = plt.subplots(2,2,figsize=(12,12))
    
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
    ax = axs[-1]
    for t in time_steps:
        V = analytical_solution(V_t,X,t,x0)
        ax.plot(X,V,label=f't={t:.2f}s')
        ax.set_xlabel('Position [m]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title('Analytical Solution')
        ax.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('test.png')

# dx1 = 0.01
dt_explicit = dx**2/Nx*10
# alpha = dt_explicit/(2*dx**2)
# print(alpha)
V_explicit = explicit_euler(V0,dx,dt_explicit,Nx,Nt)
V_implicit = implicit_euler(V0,dx,dt,Nx,Nt) 
# V_implicit = test_implicit_euler(V0,dx,dt,Nx,Nt) 
V_crank_nicolson = crank_nicolson(V0,dx,dt,Nx,Nt)
V_t = 0.2
# V_analytical = analytical_solution(V_t,X,T,x0)
time_steps = [T[1],T[25],T[50],T[450],T[900]]
# time_steps = [int(Nt/4),int(Nt/2),int(3*Nt/4),Nt-1]
# time_steps = [T[int(Nt/4)],T[int(Nt/2)],T[int(3*Nt/4)],T[Nt-1]]
# time_steps = [T[int(Nt/8)],T[int(Nt/4)],T[int(Nt/2)],T[Nt-1]]
# plot_method(V_explicit,X,time_steps,'Explicit Euler')
# plot_analytical(X,time_steps,'Figures/analytical')

plot_methods(X,V_explicit,V_implicit,V_crank_nicolson,time_steps)

