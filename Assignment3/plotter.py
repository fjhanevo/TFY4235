import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProteinPlotting:
    def __init__(self,pf,log,D=2):
        """ Initialize a ProteinFolding and Logger instance """
        self.pf = pf
        self.log = log
        self.N = pf.N
        # Dimensionality checker
        self.D = D
    
    def plot_monomer(self,sweeps):
        """ Plots proteins in both 2D and 3D """

        if self.D > 2:
            ticks = np.arange(1,self.N+3,1)
            x = self.pf.pos[:,0]
            y = self.pf.pos[:,1]
            z = self.pf.pos[:,2]
            
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111,projection='3d')
            ax.plot(x,y,z,marker='o',markersize=8,markerfacecolor='blue',
                    color = 'skyblue',linewidth=4)
            
            # Highlight first monomer
            ax.plot(x[0],y[0],z[0],marker='o',markersize=8, color='green',label='First monomer')
            
            # Highlight last monomer
            ax.plot(x[-1],y[-1],z[-1],marker='o',markersize=8, color='red',label='Last monomer')
            
            # Annotate types on the plot
            for i, t in enumerate(self.pf.types):
                ax.text(x[i]+0.065,y[i]+0.065,z[i]+0.065, f'{t}',color='black',fontsize=12)
            
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)
            plt.title(f'3D protein with randomly assigned types. Sweep(s) = {sweeps}')
            ax.grid(True)
            plt.axis('equal')
            ax.legend()
            plt.tight_layout()
            plt.show()
            
        else:
            x = self.pf.pos[:,0]
            y = self.pf.pos[:,1]

            plt.figure(figsize=(8,6))
            plt.plot(x,y,marker='o',markersize=8,markerfacecolor='blue',
                    color='skyblue',linewidth=4)

            # Highlight first monomer
            plt.plot(x[0],y[0],marker='o',markersize=8, color='green',label='First monomer')

            # Highlight last monomer
            plt.plot(x[-1],y[-1],marker='o',markersize=8, color='red',label='Last monomer')

            # Annotate types on the plot
            for i, t in enumerate(self.pf.types):
                plt.annotate(str(t),(x[i]+0.065,y[i]+0.065))

            plt.title(f'2D protein with randomly assigned types. Sweep(s) = {sweeps}')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.show()


    def plot_im(self):
        """ Plots an instance of the interaction matrix """
        plt.figure(figsize=(8,6))
        plt.imshow(self.pf.int_mat,cmap='inferno')
        plt.colorbar(label='Interaction Energy')
        plt.title(f'Monomer-Monomer Interaction Energy Matrix, Dimension = {self.D}D')
        # plt.xlabel('Monomer type')
        # plt.ylabel('Monomer type')
        plt.show()

    def plot_data(self,sweeps):
        """ Plots energy, e2e and RoG as a function of MC steps """
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        plt.plot(self.log.data['energy'], label='Energy')
        plt.title(f'Energy for Sweep(s) = {sweeps}, Dimension={self.D}D')
        plt.xlabel('MC Steps')
        plt.ylabel('Energy')
        plt.grid(True)
        plt.legend()

        plt.subplot(312)
        plt.plot(self.log.data['end_to_end-distance'], label='End-to-End Distance')
        plt.title(f'End-to-End Distance for Sweep(s) = {sweeps}, Dimension={self.D}D')
        plt.xlabel('MC Steps')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.legend()

        plt.subplot(313)
        plt.plot(self.log.data['radius_of_gyration'], label='RoG')
        plt.title(f'Radius of Gyration for Sweep(s) = {sweeps}, Dimension={self.D}D')
        plt.xlabel('MC Steps')
        plt.ylabel('RoG')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_e_rog(self):
        """ Plots energy and RoG for different chain lengths and temperatures """
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        plt.plot(self.log.data['energy'], label='Energy')
        plt.title(f'Energy for N = {self.N} chain length, Dimension={self.D}, T = {self.pf.temp}')
        plt.grid(True)
        plt.legend()

        plt.subplot(212)
        plt.plot(self.log.data['radius_of_gyration'], label='RoG')
        plt.title(f'Rog for N = {self.N} chain length, Dimension={self.D}, T = {self.pf.temp}')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def plot_average(self,key):
        """ Plots average values """
        plt.figure(figsize=(12,8))
        # plt.subplot(211)
        plt.plot(self.log.get_average(key),label = f'Average {key}')
        plt.xlabel('Steps')
        plt.legend()

        # plt.subplot(212)
        # plt.plot(self.log.get_average('radius_of_gyration'),label='Average RoG')
        # plt.xlabel('Steps')
        # plt.legend()
        #
        plt.tight_layout()
        plt.show()



 
