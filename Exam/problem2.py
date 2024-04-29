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

def disjointed_network_analyzation(N1,N2):
    """ 
    Performs the computational study on a disjointed network.
    Calculates the eigenvectors and eigenvalues of the disjointed
    transformation matrix.
    """

    T = gen_disjoint_T_matrix(N1,N2)
    steps = 100
    find_eigenvalues(T,steps)


def check_methods(N,T):
    """ Direct inversion method for solving solving V(t) """
    # Import necessary tools
    import time
    from scipy.linalg import lu_factor, lu_solve
    from scipy.sparse.linalg import cg

    # Define intial state vector 
    V_t = np.array([1/i for i in range(1,N+1)])

    # Method 1: Direct Inversion
    t0 = time.time()
    T_inv = np.linalg.inv(T)
    V_t_dt = np.dot(T_inv,V_t)
    V_t_5dt = np.linalg.matrix_power(T_inv,5) @ V_t
    time_direct = time.time()-t0
    
    # Method 2: LU Decomposition
    t0 = time.time()
    lu, piv = lu_factor(T)
    V_t_dt_lu = lu_solve((lu,piv),V_t)
    V_t_5dt_lu = V_t_dt_lu.copy()
    for _ in range(4):
        V_t_5dt_lu = lu_solve((lu,piv), V_t_5dt_lu)
    time_lu = time.time() - t0

    # Method 3: Conjugate Gradient
    t0 = time.time()
    V_t_dt_cg, _ = cg(T,V_t,tol=1e-10)
    
    # Repeatedly apply CG to simulate t-5*dt
    V_t_5dt_cg = V_t_dt_cg.copy()
    for _ in range(4):
        V_t_5dt_cg, _ = cg(T,V_t_5dt_cg,tol=1e-10)
    time_cg = time.time() - t0

    # Print results
    print('Direct Inverion Time:', time_direct)
    print('V(t-dt) by Direct Inversion:', V_t_dt)
    print('V(t-5dt) by Direct Inversion:', V_t_5dt)

    print('LU Decomposition Time:', time_lu)
    print('V(t-dt) by LU Decomposition:', V_t_dt_lu)
    print('V(t-5dt) by LU Decomposition:', V_t_5dt_lu)

    print('Conjugate Gradient Time:', time_cg)
    print('V(t-dt) by Conjugate Gradient:', V_t_dt_cg)
    print('V(t-5dt) by Conjugate Gradient:', V_t_5dt_cg)

N = 21
T = gen_T_matrix(N)
# check_methods(N,T)
N1 = 11
N2 = 10
disjointed_network_analyzation(N1,N2)
# find_eigenvalues(T,100)
