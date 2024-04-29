import numpy as np

""" 
TFY4235 Computational Physics Exam 2024.
Problem 2
"""

def gen_T_matrix(N):
    """
    Generates the transformation matrix for N nodes.
    Iterates over each node to set up the connection
    and connects to the four neihgbors.
    """

    T = np.zeros((N,N))

    for i in range(N):
        # Connect to preceeding nodes
        T[i][(i-1)%N] = 1/4
        T[i][(i-2)%N] = 1/4
        # Connect to following nodes
        T[i][(i+1)%N] = 1/4
        T[i][(i+2)%N] = 1/4

    return T

def simulate_network_evolution(T,initial_state, steps):
    """
    Simulates the evolution of the network
    Uses the transform matrix T, and accepts arbitrary
    initial states
    """
    
    # Set initial condition
    V_t = initial_state.copy()

    # Empty list to track evolution
    evolution = []

    # Iterate over time steps
    for _ in range(steps):
        # Update state vector by applying transform matrix
        V_t = T @ V_t

        # Log the new state
        evolution.append(V_t)

    return np.array(evolution)

def power_iteration(T, steps,eps=1e-10):
    """ Performs the power iteration algorithm to find the largest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T.shape[1])


    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = T @ b_k

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

        # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
        eigval = np.dot(np.transpose(b_k), T @ b_k) / (np.transpose(b_k) @b_k )
        eigvec = b_k
        
        return eigval, eigvec
# !!!!
def inverse_power_iteration(T, steps,eps=1e-10):
    """ Performs the inverse power iteration algorithm to find the smallest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T.shape[1])

    # Regularization to avoid singular matrix
    T_inv = np.linalg.inv(T+eps*np.eye(T.shape[0]))

    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = T_inv @ b_k

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

        # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
        eigval = np.dot(b_k.T, T @ b_k) / (b_k.T@b_k)
        eigvec = b_k
        
        return eigval, eigvec


N = 21
T = gen_T_matrix(N)

