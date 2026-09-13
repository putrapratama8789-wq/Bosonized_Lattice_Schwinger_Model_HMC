import numpy as np
import pandas as pd
import os
import math
from datetime import datetime
from scipy._lib._util import check_random_state
from tqdm import tqdm
from Config import *



class HMC ():

    def __init__(self, func, grad, n_prod, n_therm, kappa, 
                 epsilon, n_chain = None, rng_seed  = None,  verbose = False):


        # The function and its gradient must be callable objects.
        if callable(func):

            # Assign the function to the object.
            self.func = func
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Function {func} must be callable.")


        if callable(grad):

            # Assign the function to the object.
            self.grad = grad
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Function {grad} must be callable.")


        self._options = dict()

        self._options["n_samples"] = int(n_prod)
        self._options["n_omitted"] = int(n_therm)

        self._options["kappa"] = int(kappa)
        self._options["epsilon"] = float(epsilon)

        self._options["rng_seed"] = check_random_state(rng_seed)


        if n_chain :

            self._options["n_chain"] = int(abs(n_chain))

        else :

            #Default
            self._options["n_chain"] = 1
        
        if isinstance (verbose, bool):

            self._options["verbose"] = verbose

        else :
            raise TypeError(f"{self.__class__.__name__} :" f"verbose must be boolean")


        self._stats = None

    #________



    @property
    def n_prod (self) :

        return self._options["n_prod"]

    @n_prod.setter
    def n_prod (self, new_value) :

        if isinstance (new_value, int) :

            if new_value > 1 :

                self._options["n_prod"] = int(new_value)

            else :
                raise TypeError(f"{self.__class__.__name__} :"  
                                f"number of configuration should be positive")

        else :
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of configuration should be integer")



    @property
    def n_therm (self) :

        return self._options["n_therm"]


    @n_therm.setter
    def n_therm (self, new_value) :


        if isinstance (new_value, int) :

            if new_value > 1 :

                self._options["n_therm"] = int(new_value)

            else :
                raise TypeError(f"{self.__class__.__name__} :"  
                                f"number of configuration should be positive")

        else :
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of configuration should be integer")


    @property
    def n_chain (self) :

        return self._options["n_chain"]

    @property
    def verbose (self) :

        return self._options["verbose"]


    #########
    #### Gap isi setter and getter
    #########

    
    @staticmethod
    def leapfrog(phi, pi_mom, grad, epsilon, kappa, *args):
        pi_mom += 0.5 * epsilon * grad(phi, config)

        for step in range(kappa):
            phi += epsilon * pi_mom
            if step != kappa - 1:
                pi_mom += epsilon * grad(phi, config)

        pi_mom += 0.5 * epsilon * grad(phi, config)
        return phi, pi_mom

    
    @staticmethod
    def hmc_step(phi, func, grad, epsilon, kappa, *args):

        pi_old = np.random.randn(*phi.shape)
        
        phi_new = phi.copy()
        pi_new = pi_old.copy()

        phi_new, pi_new = HMC.leapfrog(phi_new, pi_new, grad, epsilon, kappa, *args)

        H_old = (func(phi, config) + 0.5*np.sum(pi_old**2))
        H_new = (func(phi_new, config) + 0.5*np.sum(pi_new**2))

        dH = H_new - H_old
        # Metropolis Accept/Reject
        if dH < 0 or np.random.rand() < np.exp(-dH):
            return phi_new, True, dH
        else:
            return phi, False, dH
    

    #Thermalization + Epsilon autocalibration (we target 70% acceptance rate)
    @staticmethod
    def therm (phi, func, grad, epsilon, kappa, n_therm, delta = 0.7, eta = 1.0, *args) :

        for i in tqdm(range(1, n_therm + 1), desc="Thermalizing", disable=False):

            phi, _, dH = HMC.hmc_step(phi , func,  grad, epsilon, kappa, *args)

            alpha = min(1, np.exp(-dH))
            decay = 1 / math.sqrt(i)

            log_eps = math.log(epsilon) + decay * eta * (alpha - delta)

            epsilon = math.exp(log_eps)

        return phi, epsilon





#main function and gradient

def action(phi, config):

    mg = config["mg"]
    ag = config["ag"]
    pref = config["pref"]
    theta = config["theta"]

    phi_nplus = np.roll(phi, -1, axis=0)
    phi_tplus = np.roll(phi, -1, axis=1)

    S_kin = 0.5 * np.sum((phi_nplus - phi)**2 + (phi_tplus - phi)**2)
    S_mass = ((ag)**2)/(2 * np.pi) * np.sum((phi + theta/(2*np.sqrt(np.pi)))**2)
    S_ferr = (-1) * pref * (mg) * np.sum(np.cos(2.0 * np.sqrt(np.pi) * phi))

    return S_kin + S_ferr + S_mass

def force(phi, config):

    mg = config["mg"]
    ag = config["ag"]
    pref = config["pref"]
    theta = config["theta"]

    phi_nplus = np.roll(phi, -1, axis=0)
    phi_xminus = np.roll(phi, 1, axis=0)
    phi_tplus = np.roll(phi, -1, axis=1)
    phi_tminus = np.roll(phi, 1, axis=1)

    F_neighbors = phi_nplus + phi_xminus + phi_tplus + phi_tminus

    F_kin = 4*phi - F_neighbors
    F_mass = ((ag)**2)/(np.pi) * (phi + theta/(2*np.sqrt(np.pi)))
    F_ferr = pref * (2.0*np.sqrt(np.pi)) * (mg) * np.sin(2*np.sqrt(np.pi)*phi)

    return (-1) * (F_kin + F_ferr + F_mass)


def chiral_condensate(phi, config):

    gamma = config["gamma"]
    UV_val = config["UV_val"]
    g = config["g"]

    cos_val = np.mean(np.cos(2*np.sqrt(np.pi)*phi))
    sigma_val = -(np.exp(gamma)/(2.0*(np.pi)**(3/2))) * UV_val * g * cos_val
    return sigma_val


def run_simulation(config, label="Run"):

    Lx = config["Lx"]
    Lt = config["Lt"]
    ag = config["ag"]
    
    current_fracT = Lt * ag
    print(f"\n=== Starting: {label} ===")
    
    # Initialize Field
    phi = np.random.randn(Lx, Lt)
    
    # Thermalization 

    phi, epsilon_new = HMC.therm(
        phi, 
        action, 
        force, 
        config["epsilon"], 
        config["kappa"], 
        config["n_therm"]
        )

    # Production
    data_list = []
    prod_acc = 0
    
    for i in tqdm(range(1, config["n_prod"] + 1), desc="Production"):
        phi, accepted, dH = HMC.hmc_step(phi, action, force, epsilon_new, config["kappa"])
        if accepted: prod_acc += 1

        acc_prob = np.exp(-dH)
        
        sigma = chiral_condensate(phi, config)

        # Build the data dictionary 
        step_data = {
            "run_label": label,
            "trajectory": i,
            "sigma": sigma,
            "accepted": accepted,
            "dH": dH,
            "acc_prob" : acc_prob,
            "g_over_T": current_fracT,
            "epsilon_new" : epsilon_new
        }
        
        # Merge all config variables
        step_data.update(config)
        
        data_list.append(step_data)
        
    df = pd.DataFrame(data_list)
    final_acc_rate = prod_acc / config["n_prod"]
    new_epsilon = epsilon_new
    return df, final_acc_rate, new_epsilon
    


# Saving Funtions
def save_custom_raw(dataframe, folder, config, label):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Extract for the filename
    mass = config["mg"]
    theta = config["theta"]
    
    filename = f"{folder}/raw_m{mass:.2f}_T{theta:.2f}_{label}.csv"
    dataframe.to_csv(filename, index=False)
    print(f"[Saved] Raw data: {filename}")

def save_custom_summary(summary_dict, folder, config):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Extract for the filename
    mass = config["mg"]
    theta = config["theta"]

    filename = f"{folder}/summary_m{mass:.2f}_T{theta:.2f}.csv"
    
    df = pd.DataFrame([summary_dict])
    df.to_csv(filename, index=False)
    print(f"[Saved] Run Summary: {filename}")


def jackknife_error(data_array, bin_size = 100):
    N = len(data_array)
    n_bins = N // bin_size
    if n_bins < 2: return np.mean(data_array), 0.0
    
    cutoff = n_bins * bin_size
    truncated_data = data_array[:cutoff]
    binned_data = truncated_data.reshape((n_bins, bin_size)).mean(axis=1)
    
    total_sum = np.sum(binned_data)
    jack_means = (total_sum - binned_data) / (n_bins - 1)
    
    jk_mean = np.mean(jack_means)
    jk_var = (n_bins - 1) * np.mean((jack_means - jk_mean)**2)
    jk_err = np.sqrt(jk_var)
    
    return np.mean(data_array), jk_err



if __name__ == "__main__":


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

    
    config = {
    "Lx" : 224,
    "Lt" : 224,
    "theta" : 0.0,
    "n_therm" : 200, 
    "n_prod" : 500,
    "kappa" : 100,
    "BIN_SIZE" : 500,
    "TEMP_REF" : 11.2,
    "epsilon" : 0.1,
    "mg" : 0.0,  
    "ag" : 0.1,
    "g" : 1,
    "gamma" : 0.5772156649015329
    }

    # Extract ag
    ag_current = config["ag"]
    gamma = np.euler_gamma

    # Calculate UV_val and pref
    UV_val = np.exp(2 * np.pi * feyn_cont(ag_current, N=2000))
    pref = (np.exp(gamma) / (2.0 * (np.pi)**(3/2))) * (ag_current**2) * UV_val
    config["pref"] = pref
    config["UV_val"] = UV_val

    # Run SImulation
    df, final_acc_rate, new_epsilon = run_simulation(config, label="TestRun")

    #Jackknife Error
    mean_sigma, err_sigma = jackknife_error(df["sigma"].values, bin_size=config["BIN_SIZE"])

    # Summary dictionary of the results
    run_summary = {
        "mass": config["mg"],
        "ag": config["ag"],
        "theta": config["theta"],
        "g_over_T": df["g_over_T"].values[-1],
        "Lx": config["Lx"],
        "Lt": config["Lt"],
        "mean_condensate": mean_sigma,
        "std_condensate": err_sigma,
        "n_therm": config["n_therm"],
        "n_prod": config["n_prod"],
        "kappa": config["kappa"],
        "acceptance_rate": final_acc_rate,
        "BIN_SIZE": config["BIN_SIZE"],
        "epsilon": config["epsilon"],
        "new_epsilon": new_epsilon
    }

    # Save
    output_folder_raw = "./results/raw"
    output_folder_summary = "./results/summary"

    save_custom_raw(df, output_folder_raw, config, label="TestRun")
    save_custom_summary(run_summary, output_folder_summary, config)

    

