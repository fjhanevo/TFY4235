import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Parameters
X0 = 0.         # Left side of the box
XL = 1.         # Right side of the box
N = 1000        # Interval between left and right side
m = 1.
# hbar = 6.626e-34
hbar = 1.
dx = (XL - X0) / N       # Discretization step 

x = np.linspace(X0,XL,N)  # Grid

# Defining the potential
def V(x,V0):
    # return V0 if 1/3 < x.all() < 2/3 else 0
    V=np.zeros_like(x)
    V[(x > 1./3) & (x < 2./3)] = V0
    return V



# Constructing the hamiltonian
V0 = 1e3      # for Task 3.2
# diag_elements =   V(x,V0) / dx**2
# off_diag = -1
# H = np.diag(np.full(N,diag_elements)) +\
#     np.diag(np.full(N-1, off_diag),1) +\
#     np.diag(np.full(N-1, off_diag),-1) 
# diag = np.ones(N) / dx**2 + V(x,V0)
# off_diag = np.ones(N-1) * (-0.5 / dx**2)
# H = np.diag(np.full(N,diag)) +\
#     np.diag(np.full(N-1, off_diag),1) +\
#     np.diag(np.full(N-1,off_diag),-1)
H = np.diag(np.ones(N)/dx**2 + V(x,V0))+\
    - 0.5*np.diag(np.ones(N-1)/dx**2,-1) +\
    - 0.5*np.diag(np.ones(N-1)/dx**2,1)
# Calculate eigenvalues and eigenvectors
eigenval, eigenvec = np.linalg.eigh(H)
# Normalize the eigenvectors
eigenvec /= np.sqrt(dx)

# Task 3.2
def plot_eigenvec():
    for i in range(5):
        # Plotting normalized wavefunction
        plt.plot(x,eigenvec[:,i] * np.sign(eigenvec[1,i]), 
                 label=f'$E{i+1}$')

    plt.title("Four lowest-energy wavefunctions with a barrier")
    plt.xlabel("$x/L$")
    plt.ylabel(r"$\psi(x)$")
    plt.axhline(0,color='black')
    plt.grid(True)

# Task 3.3
def init_condition(psi):
    return 1/np.sqrt(2) * (psi[0,:] + psi[1,:])

psi_0 = init_condition(eigenvec)


# Task 3.4
# Implementing the root finding function 
def f(l,V0=1e3):
    k = np.sqrt(l)
    kappa = np.sqrt(V0 - l)
    term1 = np.exp(kappa/3) * ((kappa * np.sin(k/3) + k * np.cos(k/3)))
    term2 = np.exp(-kappa/3) * ((kappa * np.sin(k/3) - k * np.cos(k/3)))
    return term1 - term2


# Plotting the root finding function
def plot_f():
    # Creating a range of eigenvalues
    l_values = np.linspace(0,V0-1,N)
    f_values = np.array([f(l) for l in l_values])
    # plt.plot(x,f(eigenval,V0),label=r'$f(\lambda)$')
    plt.plot(x,f_values, label=r'$f(\lambda)$')
    plt.title("Root-finding function")
    plt.xlabel(r"$x/L$")
    plt.ylabel(r"$f(\lambda)$")
    plt.axhline(0,color='black')
    plt.grid(True)

# Task 3.5
# Implementation of a root-finding algorithm
# Attempt to use the Newton method from the scipy library
    
# def root_finding():
#     result = []
#     l_values = np.linspace(0,V0-1,N)
#     f_values = np.array([f(l,V0) for l in l_values])
#
#     for i in f_values.shape:
#         if f[i] <
#

# print(f(eigenval))
# bisection(f,X0,XL)
# fsolve(f,[X0,XL])
plot_f()
# plot_eigenvec()
# print(root_finding())
plt.legend()
plt.show()
