from protein_folding import ProteinFolding
import numpy as np

class ProteinFolding3D(ProteinFolding):
    def __init__(self,N,T, seed=None):
        """ Initialize the base class """
        super().__init__(N,T,seed)
        # Overwrite the pos from PF class, make it 3D
        self.pos = np.zeros((N,3),dtype=int)
        # Set starting point to origin
        self.pos[0] = np.array([0,0,0])

    
    @staticmethod
    def get_moves():
        """ Defines allowed moves in 3D """
        possible_moves = [
                (1, 0, 0),   # Move right
                (-1, 0, 0),  # Move left
                (0, 1, 0),   # Move forward
                (0, -1, 0),  # Move backward
                (0, 0, 1),   # Move up
                (0, 0, -1),  # Move down
        ]
        return possible_moves

    @staticmethod
    def get_mc_moves():
        """ Defines allowed moves in 3D for the Monte Carlo step """
        possible_moves = [
                (1, 0, 0),   # Move right
                (-1, 0, 0),  # Move left
                (0, 1, 0),   # Move forward
                (0, -1, 0),  # Move backward
                (0, 0, 1),   # Move up
                (0, 0, -1),  # Move down
                #
                # x-y direction
                (1,1,0),
                (-1,1,0),
                (1,-1,0),
                (-1,-1,0),
                
                # y-z direction
                (0,1,1),
                (0,-1,1),
                (0,1,-1),
                (0,-1,-1),
                
                # x-z direction
                (1,0,1),
                (1,0,-1),
                (-1,0,1),
                (-1,0,-1)
        ]
        return possible_moves

    def place_monomers(self):
        """ 
        Randomly place monomers and check if the moves are valid
        and don't overlap
        """
        occupied_pos = set()
        occupied_pos.add(tuple(self.pos[0]))

        center= int(self.N/2)
        self.pos[0] = [center,center,center]
        for index in range(1, self.N):
            moves = self.get_moves()
            np.random.shuffle(moves)
            valid_move = False

            for move in moves:
                pos_next = self.pos[index-1] + np.array(move)
                if tuple(pos_next) not in occupied_pos:
                    self.pos[index] = pos_next
                    occupied_pos.add(tuple(pos_next))
                    valid_move = True

            if not valid_move:
                raise ValueError(f"Unable to place monomer {index}: No valid moves")



    def gen_unfolded_protein(self):
        """ Generates an unfolded protein in 3D """
        # let the protein start at the center for better visualization
        center= int(self.N/2)
        self.pos[0] = [0,center,center]
        for index in range(1,self.N):
            self.pos[index] = [index,center,center]
   


