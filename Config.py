import numpy as np


config = {
"Lx" : 224,
"Lt" : 224,
"theta" : 0.0,
"n_therm" : 200, 
"n_prod" : 1000,
"kappa" : 100,
"BIN_SIZE" : 500,
"TEMP_REF" : 11.2,
"epsilon" : 0.1,
"mg" : 0.0,  
"ag" : 0.1,
"g" : 1,
"gamma" : 0.5772156649015329
}

# Continuous Feynman propagator
def feyn_cont(ag, N=2000):
    a_mu = ag / np.sqrt(np.pi)

    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    t = x.copy()
    kx, kt = np.meshgrid(x, t, indexing='ij')

    delta_k = (4*np.sin(kx/2)**2 + 4*np.sin(kt/2)**2 + (a_mu**2))**(-1)
    
    integral = (2*np.pi)*(2*np.pi) 
    delta = (integral/(2*np.pi)**2) * delta_k

    return delta.mean()


# Extract ag
ag_current = config["ag"]
gamma = np.euler_gamma

# Calculate UV_val and pref
UV_val = np.exp(2 * np.pi * feyn_cont(ag_current, N=2000))
pref = (np.exp(gamma) / (2.0 * (np.pi)**(3/2))) * (ag_current**2) * UV_val
config["pref"] = pref
config["UV_val"] = UV_val