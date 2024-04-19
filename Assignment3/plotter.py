import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProteinPlotting:
    def __init__(self,pf,log,D=2):
        """ Initialize a ProteinFolding and Logger instance """
        self.pf = pf
        self.log = log
        # Dimensionality checker
        self.D = D
    
    def plot_monomer(self,sweeps):
        """ Plots a 2D primary structure """

        if self.D > 2:
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
            
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            ax.grid(True)
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

            plt.title(f'2D protein with randomly assigned types. Sweeps = {sweeps}')
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
        plt.title('Monomer-Monomer Interaction Energy Matrix')
        # plt.xlabel('Monomer type')
        # plt.ylabel('Monomer type')
        plt.show()

    def plot_data(self,sweeps):
        """ Plots energy, e2e and RoG as a function of MC steps """
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        plt.plot(self.log.data['energy'], label='Energy')
        plt.title(f'Energy for Sweep(s) = {sweeps}')
        plt.xlabel('MC Steps')
        plt.ylabel('Energy')
        plt.grid(True)
        plt.legend()

        plt.subplot(312)
        plt.plot(self.log.data['end_to_end-distance'], label='End-to-End Distance')
        plt.title(f'End-to-End Distance for Sweep(s) = {sweeps}')
        plt.xlabel('MC Steps')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.legend()

        plt.subplot(313)
        plt.plot(self.log.data['radius_of_gyration'], label='RoG')
        plt.title(f'Radius of Gyration for Sweep(s) = {sweeps}')
        plt.xlabel('MC Steps')
        plt.ylabel('RoG')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


 
