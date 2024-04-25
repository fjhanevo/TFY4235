import numpy as np
import matplotlib.pyplot as plt

""" 
TFY4325 Assignment 1:
Crank-Nicolson scheme for the one-dimensional diffusion equation.
"""

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

# Parameters
def crank_nicolson(N,T,bc=None):
 
    X0, XL = 0., 1.     # Domain Length
    T0, TF = 0., 1.     # Time interval
    D = 1               # Diffusion constant.
    # Initializing U to store results

    dt = (TF-T0)/T           # Time step
    dx = (XL-X0)/N    # Spatial step
    alpha = D*dt/(dx**2) # CFL number

    # Store results 
    U = np.zeros((T,N))

    # Defining initial condition
    U[0,int(N/2)] =150 


    # Get matrices
    A, B = get_matrices(N,alpha,bc)

    Xgrid = np.linspace(X0,XL,N)
    Tgrid = np.linspace(T0,TF,T)
    # Solve A*U^(n+1)=B*U^n
    for i in range(T-1):
        u = B@U[i]
        U[i+1] = np.linalg.solve(A,u)
    return U, Xgrid, Tgrid
N = 100
T = 3000
U, Xgrid,Tgrid = crank_nicolson(N,T,'N')
time_interval = [0.1,0.2,0.3,0.4,0.5]

plt.figure(figsize=(12,8))
for t in time_interval:
    index = int(t / (Tgrid[-1] /T))
    plt.plot(Xgrid,U[index],label=f't={t}')
plt.legend()
plt.grid(True)
plt.show()

