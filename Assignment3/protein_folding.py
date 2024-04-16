import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from logger import Logger
""" TFY4235 Assignment 3: Protein Folding"""

class ProteinFolding:
    def __init__(self,N, T):
        """ Initialization function"""
        self.N = N
        self.temp = T
        self.pos = np.zeros((self.N,2),dtype=int) 
        # self.pos = self.place_monomers()
        self.types = self.gen_types()
        self.int_mat = self.interaction_matrix()
        self.nn = self.find_nn()
        self.energy = self.calc_energies()


    def gen_types(self):
        """ Generates random types ranging from 1 - 20 """
        return np.random.randint(1,21,size=self.N)

    @staticmethod
    def get_moves():
        """ Defines allowed moves in 2D """
        return [(1,0),(-1,0),(0,1),(0,-1)]

    def place_monomers(self):
        """ 
        Creates an empty list to store positions.
        Randomly place monomers and check if the moves are valid
        and don't overlap
        """
        pos = np.zeros((self.N,2),dtype=int)     # Initialize positions
        occupied_pos = set()                # Track occupied positions
        occupied_pos.add(tuple(pos[0]))     # Set start to origin

        for i in range(1, self.N):
            moves = self.get_moves()
            np.random.shuffle(moves)       # Randomize order of trial
            valid_move = False

            for move in moves:
                pos_next = pos[i-1] + np.array(move)
                if tuple(pos_next) not in occupied_pos:
                    pos[i] = pos_next
                    occupied_pos.add(tuple(pos_next))
                    valid_move = True
                    break   # Stop after finding a valid move

            if not valid_move:
                # Raise an exception if an invalid move is found
                raise ValueError(f"Unable to place monomer {i}")

        self.pos = pos

    def find_nn(self):
        """ 
        Finds the nearest-neighbors for each monomer.
        Only accept the first non-covalent interaction
        """
        nn = [] # Nearest-neighbors list

        for i, pos_now in enumerate(self.pos):
            n = []
            for j, pos_next in enumerate(self.pos):
                if i != j:  # Avoid self-comparison
                    if (abs(pos_now[0] - pos_next[0]) == 1 \
                        and pos_now[1] == pos_next[1]) \
                        or (abs(pos_now[1] - pos_next[1]) == 1 \
                        and pos_now[0] == pos_next[0]):
                            # Exclude directi sequential neighbors to
                            # avoid calculating covalent bonds
                            if abs(i-j) > 1:
                                n.append(j)
                                # Only accept first non-covalent
                                break
            nn.append(n)
        return nn


    @staticmethod
    def interaction_matrix():
        """ Create interaction matrix, make sure it stays constant """
        # np.random.seed(10)
        int_mat = np.random.uniform(-4,-2,(20,20))
        # Ensure that the matrix is symmetric
        return (int_mat + int_mat.T)/2

    def calc_e2e(self):
        """ Calculates the end-to-end distance """
        return np.linalg.norm(self.pos[-1] - self.pos[0]) 

    def calc_rog(self):
        """ Calculated the radius of gyration """
        center_of_mass = np.mean(self.pos, axis=0)
        return np.sqrt(np.mean(np.sum((self.pos - center_of_mass)**2, axis=1)))

    def calc_energies(self):
        """ Calculate energy of tertiary structures """
        total_energy = 0
        for i, neighbors in enumerate(self.nn):
            for neighbor in neighbors:
                # Subtract 1 to make type index start at 0
                energy = self.int_mat[self.types[i]-1,self.types[neighbor]-1]
                total_energy += energy
        return total_energy

    def is_valid_move(self, index, move):
        """ Check if a move is valid and don't result in overlap """
        # Check for overlap
        pos_new = self.pos[index] + move

        print(f"Trying {move} for monomer {index} from {self.pos[index]} to {pos_new}")
        # Check for diagonal moves
        if abs(move[0]) == abs(move[1]) == 1:
            print("DIAGONAL!!!!!!!!!")
            return False

        if any(np.array_equal(pos_new, pos) for pos in self.pos):
            print("Move results in overlap")
            return False
        
        
        print("OK")
        return True

        # Check that position is within the bounds
    
    def metropolis_criterion(self,current_energy,new_energy):
        """ Metropolis criterion used to determine if a move should be accepted """
        dE = new_energy - current_energy
        
        if dE < 0:
            return True
        else:
            return np.random.rand() < np.exp(-dE/self.temp) 




    def perform_mc_step(self, logger):
        """
        Performs a MC step and determines if moves are possible.
        Accepts or declines new energy configurations.
        """
        # Select a random monomer
        index = np.random.randint(0,self.N-1)

        moves = self.get_moves()
        # Shuffle moves to ensure randomness
        np.random.shuffle(moves)

        current_energy = self.energy
        old_position = self.pos[index].copy()

        for move in moves:
            print(f"checking move {move} for monomer {index}")
            if self.is_valid_move(index,move):
                self.pos[index] += np.array(move)
                new_energy = self.calc_energies()

                if self.metropolis_criterion(current_energy,new_energy):
                    print("move accepted")
                    self.energy = new_energy
                    e2e = self.calc_e2e()
                    rog = self.calc_rog()

                    logger.log_all(self.energy,self.pos.copy(),e2e,rog)
                    break
                else:
                    print("Move rejected by Metropolis")
                    # Revert move
                    self.pos[index] = old_position


            
    def run_simulation(self,sweeps,logger):
        """ Runs the MC simulation """
        for _ in range(sweeps):
            self.perform_mc_step(logger)
        logger.plot_data(sweeps)
    

    def gen_unfolded_protein(self):
        """ Generates an unfolded protein """
        # Reset the position 
        self.pos = np.zeros((self.N,2),dtype=int)

        for i in range(self.N):
            self.pos[i] = [i,0]


    def plot_im(self):
        """ Plots an instance of the interaction matrix """
        plt.figure(figsize=(8,6))
        plt.imshow(self.int_mat,cmap='inferno')
        plt.colorbar(label='Interaction Energy')
        plt.title('Monomer-Monomer Interaction Energy Matrix')
        plt.xlabel('Monomer type')
        plt.ylabel('Monomer type')
        plt.show()
    
    def plot_2D(self):
        """ Plots a 2D primary structure """
        x = self.pos[:,0]
        y = self.pos[:,1]

        plt.figure(figsize=(8,6))
        plt.plot(x,y,marker='o',markersize=8,markerfacecolor='blue',
                 color='skyblue',linewidth=4)

        # Highlight first monomer
        plt.plot(x[0],y[0],marker='o',markersize=8, color='green',label='First monomer')

        # Highlight last monomer
        plt.plot(x[-1],y[-1],marker='o',markersize=8, color='red',label='Last monomer')

        # Annotate types on the plot
        # for i, t in enumerate(self.types):
        #     plt.annotate(str(t),(x[i],y[i]))

        plt.title('2D Primary structure with randomly assigned types')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()
    
# if __name__ == 'main':

def task1_5():
    # Initialize a protein with 15 Monomers and T = 10
    p = ProteinFolding(15,10)
    
    # Initalize a logger instance
    logger = Logger()

    # Make sure its unfolded
    p.gen_unfolded_protein()
    p.plot_2D()
    p.plot_im()
    # Perform X = 1,10,100 sweeps and get e2e,rog and energy
    p.run_simulation(1,logger)
    p.run_simulation(10,logger)
    p.plot_im()
    # Check if the plot is updated
    p.plot_2D()
    p.run_simulation(100,logger)
    p.plot_2D()
    p.plot_im()

task1_5()
