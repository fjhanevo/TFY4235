import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Logger:
    def __init__(self):
        self.data = {
                'energy': [],
                'positions': [],
                'end_to_end-distance': [],
                'radius_of_gyration': []
        }

    def log_all(self, energy, positions, e2e, rog):
        """ Logs all properties at once """
        self.data['energy'].append(energy)
        self.data['positions'].append(positions)
        self.data['end_to_end-distance'].append(e2e)
        self.data['radius_of_gyration'].append(rog)

    def get_data(self,key):
        """ Retrieve a specific logged data """
        return self.data.get(key,[])

    def plot_data(self,sweeps):
        """ Plots energy, e2e and RoG as a function of MC steps """
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        plt.plot(self.data['energy'], label='Energy')
        plt.title(f'Energy vs. {sweeps} Sweep(s)')
        plt.xlabel('MC Steps')
        plt.ylabel('Energy')
        plt.grid(True)
        plt.legend()

        plt.subplot(312)
        plt.plot(self.data['end_to_end-distance'], label='End-to-End Distance')
        plt.title(f'End-to-End Distance vs. {sweeps} Sweep(s)')
        plt.xlabel('MC Steps')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.legend()

        plt.subplot(313)
        plt.plot(self.data['radius_of_gyration'], label='RoG')
        plt.title(f'Radius of Gyration vs. {sweeps} Sweep(s)')
        plt.xlabel('MC Steps')
        plt.ylabel('RoG')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


    def export_data(self,filename):
        """ Export logged data to a Jéson file """
        with open(filename, 'w') as file:
            json.dump(self.data, file)

    def get_statistics(self):
        """ Compute and return statsistics for the logged data """
        stats = {}
        for key, values in self.data.items():
            stats[key] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'variance': np.var(values)
            }
        return stats





