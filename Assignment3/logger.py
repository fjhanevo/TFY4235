import json
import numpy as np

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

    def get_average(self,key):
        """ Return average value of specified key """
        if key not in self.data:
            raise ValueError(f'{key} not available')
        else:
            return np.mean(self.data[key])

    def get_energy(self):
        return np.mean(self.data['energy'])


    def get_data(self,key):
        """ Retrieve a specific logged data """
        return self.data.get(key,[])

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





