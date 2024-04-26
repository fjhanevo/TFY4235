import numpy as np

""" 
TFY4325 Assignment 1:
Crank-Nicolson scheme for the one-dimensional diffusion equation.
"""
#

# Constructing matrices
def get_matrices(N, alpha, bc=None):
    """
    Initializes the A and B matrix used for the Crank-Nicolson scheme. 
    Applies absorbant boundary conditions by default (Dirichlet), 
    else it applies reflective boundary conditions (Neumann).
    """
    diag = np.ones((N))

    A = (-alpha/2) * np.diag(diag[1:], -1) +\
        (-alpha/2) * np.diag(diag[1:],1) +\
        (1+alpha)*np.diag(diag)

    B = (alpha/2) * np.diag(diag[1:], -1) +\
        (alpha/2) * np.diag(diag[1:],1) +\
        (1-alpha) * np.diag(diag)
    
    # Check for Neumann bc's, Dirichlet bc's by default
    if bc == 'N':
        A[0,0] = 1 + alpha/2
        A[-1,-1] = 1 + alpha/2
        B[0,0] = 1 - alpha/2
        B[-1,-1] = 1 - alpha/2

    return A,B

def crank_nicolson(N,T,alpha,U0,bc=None):
    """ 
    Crank-Nicolson scheme for solving the 1D 
    diffusion equation. 
    """
 
    # Initialize U to store results 
    U = np.zeros((T,N))

    # Defining initial condition, approximating the Dirac delta fn
    U[0,int(N/2)] = U0

    # Get matrices
    A, B = get_matrices(N,alpha,bc)

    # Solve A*U^(n+1)=B*U^n
    for i in range(T-1):
        u = B@U[i]
        U[i+1] = np.linalg.solve(A,u)
    
    return U

