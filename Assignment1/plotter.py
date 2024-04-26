import numpy as np
import matplotlib.pyplot as plt
from crank_nicolson import crank_nicolson

def plot_c_and_n(U,T,Xgrid,Tgrid):
    
    time_interval = [0.1,0.2,0.3,0.4,0.5]
    plt.figure(figsize=(12,8))
    for t in time_interval:
        index = int(t / (Tgrid[-1] /T))
        plt.plot(Xgrid,U[index],label=f't={t}')
    plt.legend()
    plt.grid(True)
    plt.show()

