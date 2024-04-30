import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid 

""" 
Implementation of a finite-difference scheme for the
1D particle in a box.
"""

# Parameters
X0 = 0.         # Left side of the box
XL = 1.         # Right side of the box
N = 1000        # Interval between left and right side

dx = (XL - X0) / N       # Discretization step 

x = np.linspace(X0,XL,N)  # Grid

""" Potensialet er ikke så viktig her siden det er 0"""
# Defining the potential
# def V(x):
#     return 0 if X0 < x < XL else np.inf

# Constructing the Hamiltonian
H = -1*np.diag(np.ones(N-1),-1) + 2*np.diag(np.ones(N)) -1*np.diag(np.ones(N-1), 1)

# H = np.diag(np.ones(N)/dx**2)+\
#     - 0.5*np.diag(np.ones(N-1)/dx**2,-1) +\
#     - 0.5*np.diag(np.ones(N-1)/dx**2,1)
# Constructing the eigenvectors
val, psi = np.linalg.eigh(H)

# Defining the exact wavefunction as given from the assignment
def psi_exact(n):
    return np.sqrt(2)*np.sin(n*np.pi*x)

# Task 2.4:
# Plotting the four lowest energy wavefunctions
def plot_SE():
    for i in range(4):
        # Normalizing and plotting the numerical wavefunction
        plt.plot(x, psi[:,i]/np.linalg.norm(psi[:,i]) *np.sign(psi[1,i]), label= f'$E{i+1}$')
    for i in range(4):
        # Normalizing and plotting the exact wavefunction
        plt.plot(x, psi_exact(i+1)/np.linalg.norm(psi_exact(i+1)),
                label ='$E_{ex}$'f'${i+1}$', linestyle='dashed')
    plt.title("Four lowest-energy wavefunctions",fontsize=14)
    plt.xlabel("$x/L$")
    plt.ylabel(r"$\psi(x)$")
    plt.axhline(0, color='black')
    plt.grid(True)

# Task 2.5: 
# Plotting computed eigenvalues against exact eigenvalues
""" Noe rart med denne, sjekk H tror jeg"""
def plot_eigenvals():
    computed_vals = val / (2*dx**2) # Normalizing 
    num_eigenvals = len(computed_vals)

    n_values = np.arange(1,num_eigenvals+1)
    exact_vals = [(np.pi*n)**2 for n in n_values]  
    plt.title("Computed and Exact eigenvalues")
    plt.xlabel("$n$")
    plt.ylabel(r"$\lambda_n = \frac{2 m L^2 E_n}{\hbar}$",fontsize=14)
    plt.plot(n_values,computed_vals,label="Numerical")
    plt.plot(n_values,exact_vals,label="Exact")

# Normalizing the wavefunction for easier further use
def normalize_psi(psi):
    psi_squared = np.abs(psi)**2

    norm_squared = np.sum(psi_squared)
    norm = np.sqrt(norm_squared)
    normalized_psi = psi / norm

    return normalized_psi


# Task 2.8: fn numerically solves the expansion coefficients
def compute_expansion(psi,psi_0):
    num_states = len(psi)
    alpha_n = np.zeros(num_states)  # Array to store expansion coefficients
    for i in range(num_states):
        # Compute the inner product using the trapezoidal rule
        alpha_n[i] = trapezoid(psi[:,i]*psi_0,x)

    return alpha_n

# Task 2.9: fn to check orthogonality
# Fortsett på denne, anta at forrige er korrekt,
def inner_product(psi,psi_0):
    num_states = len(psi) 
    # inner_product = np.zeros((num_states,num_states))
    inner_product = np.zeros(num_states)
    #
    # for n in range(num_states):
    #     for m in range(num_states):
    #         inner_product[n,m] = np.dot(psi[:,n],psi_0)
    # return inner_product
    for n in range(num_states):
        inner_product[n] = np.dot(psi[:,n],psi_0)
    return inner_product
# Task 2.10: Testing with exact solution
def task_2_10():
    psi_0 = psi_exact(1)
    psi_norm = normalize_psi(psi)
    a_n = inner_product(psi_norm,psi_0)
    for i in range(len(a_n)):
        if np.abs(a_n[i]) >=10e-3:
            print(a_n[i])

# task_2_10()
# Task 2.11: Testing with new init condition
def psi_delta():
    # Finding index closet to x = 1/2
    index = int(round(0.5/dx))
    psi_0 = np.zeros(N)
    psi_0[index] = 1.0 / np.sqrt(dx)
    return psi_0

def task_2_11():
    psi_0 = psi_delta()
    psi_norm = normalize_psi(psi)
    a_n = inner_product(psi_norm,psi_0)
    for i in range(len(a_n)):
        if np.abs(a_n[i]) >=10e-3:
            print(a_n[i])
# task_2_11()

plt.figure(figsize=(6,4.5))
# plot_SE()
plot_eigenvals()
# print(compute_expansion(norm_eigenfunc))
plt.legend()
plt.show()
