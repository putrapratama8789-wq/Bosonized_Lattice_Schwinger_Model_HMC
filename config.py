import numpy as np
import math
from numpy import sin, pi


#Fixed lattice spacing and gauge coupling
a = 0.1
g = 1.0
ag = a*g

#Lattice dimensions
#for Lx*ag = 22,4
Lx = int(math.ceil(22.4 / ag))

# Configuration Parameters
N_THERM = 5000
N_TRAJ = 50000
N_STEPS = 100
LT_REF = 448  #Low temp at g/T = 11,2 for a*g = 0.025 ; LT_REF = 448
BIN_SIZE = 500
TEMP_REF = 11.2

# list temp values (g/T)
fracT_values = np.arange(0.2, 2.2, 0.2).tolist() + [11.2]

# List Mass values (m/g)
#mass_values = [0.0625, 0.125, 0.25, 0.5, 1.0]

#For Fixed mass value, activate cells below and deactivate value on top
FIXED_MASS = 0.0625

#List of vacuum angle values (theta)
theta_input = np.linspace(0, np.pi , 13)

#Fixed vacuum angle, diactivate if you want to loop over theta values
#THETA = 0.0


gamma = 0.5772156649015329

#continous feynman propagator
def feyn_cont(ag, N):
    a_mu = ag/np.sqrt(pi)

    x = np.linspace(-pi, pi, N, endpoint=False)
    t = x.copy()

    kx,kt = np.meshgrid(x, t, indexing='ij')

    delta_k = (4*sin(kx/2)**2 + 4*sin(kt/2)**2+ (a_mu**2))**(-1)
    integral = (2*pi)*(2*pi)
    delta = (integral/(2*pi)**2) *delta_k

    feyn_prop = delta.mean()

    return feyn_prop



#uv divergence subtraction value
UV_val = np.exp(2 * np.pi * feyn_cont(ag, N=2000))

#feynman prefactor
pref = (np.exp(gamma) / (2.0*(np.pi)**(3/2))) * (ag**2) * UV_val



