import numpy as np
import matplotlib.pyplot as plt

""" 
TFY4325 Assignment 1:
Crank-Nicolson scheme for the one-dimensional diffusion equation.
"""

# Parameters
N = 100             # Spatial Grid
X0, XL = 0., 1.     # Domain Length
T = 1.              # Max time
NT = 100            # Time steps
D = 1               # Diffusion constant.

dt = 1e-3           # Time step
dx = XL / (N + 1)    # Spatial step
# alpha = D*dt/(dx**2) # CFL number
alpha=1

# Spatial grid
X = np.linspace(X0,XL,N+1)
   
# Constructing matrices
diag = np.ones((N+1))
A = (-alpha/2) * np.diag(diag[1:], -1) +\
    (-alpha/2) * np.diag(diag[1:],1) +\
    (1+alpha)*np.diag(diag)

B = (alpha/2) * np.diag(diag[1:], -1) +\
    (alpha/2) * np.diag(diag[1:],1) +\
    (1-alpha)*np.diag(diag)

# Initializing U to store results
U = np.zeros((N+1,N+1))

# Applying boundary conditions
U[:,0] = 0
U[:,-1] = 0
