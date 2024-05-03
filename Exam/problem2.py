import numpy as np
import matplotlib.pyplot as plt
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

    # Fill the matrix according to the rules
    for i in range(N):
        # Define indicies of the four neighbors
        neighbors = [(i - 2) % N, (i - 1) % N, (i + 1) % N, (i + 2) % N]

        # Assign values to neighbors
        for neighbor in neighbors:
            T[i,neighbor] = 0.25
    return T

def graph_matrix(T):
    """ Creates a graph of the transformation matrix T for visualization purposes """
    import networkx as nx
    # Create a graph from the transformation matrix
    G = nx.Graph()
    for i in range(N):
        for j in range(N):
            if T[i, j] > 0:
                G.add_edge(i+1, j+1)  # +1 to have nodes 1-indexed

    # Draw the graph
    pos = nx.circular_layout(G)  # Circular layout for better visualization
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=12, edge_color='grey')
    # plt.savefig('Figures/n21_t_graph.png')
    plt.show()

def simulate_network_evolution(N, T, steps):
    """
    Simulates the evolution of the network
    Uses the transform matrix T, and accepts arbitrary
    initial states
    """
    
    # Set initial condition
    V = np.random.rand(N) 

    # Normalize
    V = V / np.linalg.norm(V)

    # Empty list to track evolution
    evolution = np.zeros((steps+1,N))
    evolution[0] = V 

    # Iterate over time steps
    for step in range(1,steps+1):
        # Update state vector by applying transform matrix
        V = T @ V

        # Log the new state
        evolution[step]=V

    plt.figure(figsize=(14, 8))
    for i in range(N):
        plt.plot(range(steps + 1), evolution[:, i], label=f'Node {i+1}')
    plt.xlabel('Time Steps',fontsize=18)
    plt.ylabel('State',fontsize=18)
    plt.legend(loc='upper right', ncol=3,fontsize=15)
    plt.grid(True)
    plt.savefig('Figures/network_evolution.png')
    plt.show()

def power_iteration(T_mat, steps,tol=1e-10):
    """ Performs the power iteration algorithm to find the largest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T_mat.shape[1])

    b_k1_norm = 0

    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = np.dot(T_mat, b_k)

        # Calculate the norm
        b_k1_norm_new = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm_new

        # Convergence check
        if np.abs(b_k1_norm_new - b_k1_norm) < tol:
            break
        b_k1_norm = b_k1_norm_new

    # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
    eigval = np.dot(b_k.T, np.dot(T_mat,b_k)) / np.dot(b_k.T,b_k) 
    eigvec = b_k
        
    return eigval, eigvec

def inverse_power_iteration(T_mat, steps,tol=1e-10,eps=1e-10):
    """ Performs the inverse power iteration algorithm to find the smallest eigenvalues """
    # Select a random (uniform) vector
    b_k = np.random.rand(T_mat.shape[1])
    b_k1_norm = 0

    # Regularization to avoid singular matrix
    T_inv = np.linalg.inv(T_mat+eps*np.eye(T_mat.shape[0]))

    for _ in range(steps):
        # Calculate the matrix by dot product
        b_k1 = np.dot(T_inv, b_k)

        # Calculate the norm
        b_k1_norm_new = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm_new

        # Convergence check
        if np.abs(b_k1_norm_new - b_k1_norm) < tol:
            break

        b_k1_norm = b_k1_norm_new


    # Use Rayleigh quotient to get associated eigenvectors and eigenvalues
    eigval = np.dot(b_k.T, np.dot(T_mat, b_k)) / np.dot(b_k.T,b_k) 
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



def gen_disjoint_T_matrix(N1, N2):
    """
    Generates a transformation matrix for two disjoint networks.
    One network will have N1 nodes, and the other will have N2 nodes.
    """
    N = N1 + N2
    T = np.zeros((N, N))

    # Fill the matrix for the first network
    for i in range(N1):
        neighbors = [(i - 2) % N1, (i - 1) % N1, (i + 1) % N1, (i + 2) % N1]
        for neighbor in neighbors:
            T[i, neighbor] = 0.25

    # Fill the matrix for the second network
    offset = N1
    for i in range(N2):
        neighbors = [(i - 2) % N2, (i - 1) % N2, (i + 1) % N2, (i + 2) % N2]
        for neighbor in neighbors:
            T[offset + i, offset + neighbor] = 0.25

    return T
    
def disjointed_network_analyzation(N1,N2):
    """ 
    Performs the computational study on a disjointed network.
    Calculates the eigenvectors and eigenvalues of the disjointed
    transformation matrix.
    """

    T = gen_disjoint_T_matrix(N1,N2)
    print(T)
    # steps = 100
    # find_eigenvalues(T,steps)
    eigvals,_ = np.linalg.eig(T)
    # index = eigvals.argsort()[::-1]
    # eigvals = eigvals[index]

    tol = 1e-8
    eigvals_to_one = np.sum(np.isclose(eigvals,1,atol=tol))
    print('Numpy eigvals: ', eigvals)
    print()
    print('Eigvals close to one: ',eigvals_to_one)



def check_methods(N,T):
    """ 
    Compares the efficiency of 3 solvers of linear equations.
    The methods used are:
    Direct Inversion
    LU Decomposition
    Conjugate Gradient
    """
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

    print('LU Decomposition Time:', time_lu)

    error_dt = calculate_errors(V_t_dt_lu, V_t_dt)
    error_5dt = calculate_errors(V_t_5dt_lu, V_t_5dt)

    error_cg_dt = calculate_errors(V_t_dt_lu, V_t_dt_cg)
    error_cg_5dt = calculate_errors(V_t_5dt_lu, V_t_5dt_cg)

    # Print results
    print('Direct Inversion Time:', time_direct)
    print('Direct Inversion Errors for V(t-dt):', error_dt)
    print('Direction inversion Errors for V(t-5dt):', error_5dt)

    print('Conjugate Gradient Time:', time_cg)
    print('Conjugate Gradient Errors for V(t-dt):', error_cg_dt)
    print('Conjugate Gradient Errors for V(t-5dt):', error_cg_5dt)


def calculate_errors(ref, comp):
    """ Calculates the error for the methods based on a reference """
    abs_error = np.abs(ref - comp)
    rel_error = abs_error / np.abs(ref)
    return np.mean(abs_error), np.mean(rel_error)

N = 21
T = gen_T_matrix(N)
check_methods(N,T)
# N1 = 11
# N2 = 10
# disjointed_network_analyzation(N1,N2)
# T = gen_disjoint_T_matrix(N1,N2)
# find_eigenvalues(T,100)
# plot_method_comparison(T,100)
# graph_matrix(T)
# simulate_network_evolution(N,T,100)
# verify_network_state(N,T,100)
