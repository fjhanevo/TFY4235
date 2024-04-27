import numpy as np
import matplotlib.pyplot as plt
from diffusion_equation import * 

def plot_c_and_n():
    """ Plots the absorbant, reflective, and unbounded solutions """


    U1 = crank_nicolson(N,T,alpha,U0)
    U2 = crank_nicolson(N,T,alpha,U0,'N') 

    time_interval = [0.1,0.2,0.3,0.4,0.5]
    plt.figure(figsize=(12,8))
    plt.subplot(1,3,1)
    for t in time_interval:
        index = int(t / (Tgrid[-1] /T))
        plt.plot(Xgrid,U1[index],label=f't={t}')
    plt.title("Absorbant Boundary Conditions")
    plt.xlabel(f"Position $[m]$")
    plt.ylabel(f"Concentration $[kg/m^3]$")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,2)
    for t in time_interval:
        index = int(t / (Tgrid[-1] /T))
        plt.plot(Xgrid,U2[index],label=f't={t}')
    plt.title("Reflective Boundary Conditions")
    plt.xlabel(f"Position $[m]$")
    plt.ylabel(f"Concentration $[kg/m^3]$")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,3)
    for t in time_interval:
        U3 = unbounded_solution(1,D,Xgrid,0.5,t)
        plt.plot(Xgrid,U3, label=f't={t}')
    plt.title("Unbounded Analytical Solution")
    plt.xlabel(f"Position $[m]$")
    plt.ylabel(f"Concentration $[kg/m^3]$")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_total_mass():
    """
    Plots the total mass over time for the
    absorbant and reflective boundary conditons
    """
    U1 = crank_nicolson(N,T,alpha,U0)
    U2 = crank_nicolson(N,T,alpha,U0,'N') 
    
    abs_mass = total_mass(U1,dx)
    refl_mass = total_mass(U2,dx)

    plt.plot(Tgrid, abs_mass,label='Absorbant boundary condition')
    plt.plot(Tgrid,refl_mass, label='Reflective boundary condition')

    plt.title("Total mass over time")
    plt.xlabel("Time $[s]$")
    plt.ylabel("Mass $[kg]$")
    plt.grid(True)
    plt.legend()
    plt.show()


