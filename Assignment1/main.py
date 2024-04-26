import numpy as np
from crank_nicolson import crank_nicolson
import plotter as plot
N = 100
T = 3000
X0, XL = 0., 1.     # Domain Length
T0, TF = 0., 1.     # Time interval
D = 1               # Diffusion constant.

dt = (TF-T0)/T           # Time step
dx = (XL-X0)/N    # Spatial step
alpha = D*dt/(dx**2) # CFL number

Xgrid = np.linspace(X0,XL,N)
Tgrid = np.linspace(T0,TF,T)
U0 = 1/dx
U = crank_nicolson(N,T,alpha,U0)
# plot_c_and_n(U,T,Xgrid,Tgrid)
plot.plot_c_and_n(U,T,Xgrid,Tgrid)
