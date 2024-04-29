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
    
    # Each node transfer its charge equally to the neighboring nodes
    for i in range(N):
        # Connect to preceeding nodes
        T[i][(i-1)%N] = 0.25 
        T[i][(i-2)%N] = 0.25 
        # Connect to following nodes
        T[i][(i+1)%N] = 0.25 
        T[i][(i+2)%N] = 0.25 

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

def power_iteration(T_mat, steps):
    """ Performs the power iteration algorithm to find the largest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T_mat.shape[1])


    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = np.dot(T_mat, b_k)

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

        # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
        eigval = np.dot(b_k.T, np.dot(T_mat,b_k)) / np.dot(b_k.T,b_k) 
        eigvec = b_k
        
        return eigval, eigvec

def inverse_power_iteration(T_mat, steps,eps=1e-10):
    """ Performs the inverse power iteration algorithm to find the smallest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T_mat.shape[1])

    # Regularization to avoid singular matrix
    T_inv = np.linalg.inv(T_mat+eps*np.eye(T_mat.shape[0]))

    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = np.dot(T_inv, b_k)

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

        # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
        eigval = np.dot(b_k.T, np.dot(T, b_k)) / np.dot(b_k.T,b_k) 
        eigvec = b_k
        
        return eigval, eigvec

def find_eigenvalues(T,steps):
    largest_eigval,largest_eigvec = power_iteration(T,steps)
    smallest_eigval,smallest_eigvec = inverse_power_iteration(T,steps)

    # Get numpys functions to calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(T)

    # Sorting eigenvalues and corresponding eigenvectors
    index = eigvals.argsort()[::-1]
    eigvals = eigvals[index]
    eigvecs = eigvecs[:,index]

    # Find the number of eigenvalues close to 1 within the tolerance
    tol = 1e-5
    eigvals_to_one = np.sum(np.isclose(eigvals,1,atol=tol))
    
    print('Largest eigenvalue (iterative):', largest_eigval)
    print('Largest eigevector (iterative):', largest_eigvec)
    print('Smallest eigenvalue (iterative):', smallest_eigval)
    print('Smallest eigenvector(iterative):', smallest_eigvec)
    print('Number of eigenvalues close to 1:', eigvals_to_one)
    print('Eigenvalues from numpy linalg library:',eigvals)

def verify_network_state(N,T):
    """
    Verifies that the final state of the network evolution is 
    independent on the inital state by using 3 different 
    initial states
    """
    init_states = {
            'uniform': np.ones(N),
            'random': np.random.rand(N),
            'single_charged': np.zeros(N)
            }

    # Charge only the first node
    init_states['single_charged'][0] = 1

    final_state = {} 

    steps = 100

    # Run the simulation for each inital state and store the final state
    for key, state in init_states.items():
        evolution = simulate_network_evolution(T,state,steps)
        # Get the final state from the evolution
        final_state[key] = evolution[-1]

    # Comparing all final states
    all_close = all(np.allclose(final_state['uniform'], final_state[other], atol=1e-2)
                    for other in ['random','single_charged'])
    
    print('Are all final states similar?', all_close)


def gen_disjoint_T_matrix(N1,N2):
    """ Generates a transfer matrix for a disjointed network """
    
    # Initialize T
    T = np.zeros((N1+N2, N1+N2))

    # First sub-network
    for i in range(N1):
        T[i][(i-1) % N1] = 0.25
        T[i][(i-2) % N1] = 0.25
        T[i][(i+1) % N1] = 0.25
        T[i][(i+2) % N1] = 0.25

    # Second sub-network, offset by N1
    for i in range(N1,N1+N2):
        T[i][(i-1) % (N1+N2)] = 0.25
        T[i][(i-2) % (N1+N2)] = 0.25
        T[i][(i+1) % (N1+N2)] = 0.25
        T[i][(i+2) % (N1+N2)] = 0.25
    
    return T


N = 21
T = gen_T_matrix(N)
verify_network_state(N,T)
