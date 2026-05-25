import numpy as np
import pandas as pd
import os
from datetime import datetime
from config import *


def action(phi, mg, theta):

    phi_nplus = np.roll(phi, -1, axis=0)
    phi_tplus = np.roll(phi, -1, axis=1)

    S_kin = 0.5 * np.sum((phi_nplus - phi)**2 + (phi_tplus - phi)**2)
    S_mass = ((ag)**2)/(2 * np.pi) * np.sum((phi + theta/(2*np.sqrt(np.pi)))**2)
    S_ferr = (-1) * pref * (mg) * np.sum(np.cos(2.0 * np.sqrt(np.pi) * phi))

    return S_kin + S_ferr + S_mass

def force(phi, mg, theta):

    phi_nplus = np.roll(phi, -1, axis=0)
    phi_xminus = np.roll(phi, 1, axis=0)
    phi_tplus = np.roll(phi, -1, axis=1)
    phi_tminus = np.roll(phi, 1, axis=1)

    F_neighbors = phi_nplus + phi_xminus + phi_tplus + phi_tminus

    F_kin = 4*phi - F_neighbors
    F_mass = ((ag)**2)/(np.pi) * (phi + theta/(2*np.sqrt(np.pi)))
    F_ferr = pref * (2.0*np.sqrt(np.pi)) * (mg) * np.sin(2*np.sqrt(np.pi)*phi)

    return (-1) * (F_kin + F_ferr + F_mass)


def leapfrog(phi, mg, theta, pi_mom, epsilon, n_steps):
    pi_mom += 0.5 * epsilon * force(phi, mg, theta)

    for step in range(n_steps):
        phi += epsilon * pi_mom
        if step != n_steps - 1:
            pi_mom += epsilon * force(phi, mg, theta)

    pi_mom += 0.5 * epsilon * force(phi, mg, theta)
    return phi, pi_mom


def hmc_step(phi, mg, theta, epsilon, n_steps):
    
    current_Lx, current_Lt = phi.shape
    
    phi_old = phi.copy()
    pi_mom = np.random.randn(current_Lx, current_Lt)
    pi_old = pi_mom.copy()

    phi_new, pi_new = leapfrog(phi_old.copy(), mg, theta, pi_mom.copy(), epsilon, n_steps)

    dH = (action(phi_new, mg, theta) + 0.5*np.sum(pi_new**2)) - (action(phi_old, mg, theta) + 0.5*np.sum(pi_old**2))
    
    # Metropolis Accept/Reject
    if dH < 0 or np.random.rand() < np.exp(-dH):
        return phi_new, True, dH
    else:
        return phi_old, False, dH

def chiral_condensate(phi):

    cos_val = np.mean(np.cos(2*np.sqrt(np.pi)*phi))
    sigma_val = -(np.exp(gamma)/(2.0*(np.pi)**(3/2))) * UV_val * g * cos_val
    return sigma_val

# DATA & ANALYSIS

def save_raw_data(dataframe, filename_prefix="schwinger_raw"):
    if not os.path.exists('data'):
        os.makedirs('data')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"data/{filename_prefix}_{timestamp}.csv"
    dataframe.to_csv(filename, index=False)
    print(f"[Saved] Raw data saved to: {filename}")
    return filename

def save_result_summary(result_dict):

    if not os.path.exists('data'):
        os.makedirs('data')
        
    filename = "data/simulation_summary_log.csv"
    
    result_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([result_dict])
    
    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False, mode='w')
    else:
        df_new.to_csv(filename, index=False, mode='a', header=False)
        
    print(f"[Saved] Experiment summary appended to: {filename}")


def run_simulation(current_Lt, mg, theta, n_therm, n_traj, epsilon, n_steps, label="Run"):
    current_fracT = current_Lt * ag

    print(f"\n=== Starting: {label} ===")
    
    # Initialize Field
    phi = np.random.randn(Lx, current_Lt)
    
    # Thermalization
    for _ in range(n_therm):
        phi, _, _ = hmc_step(phi, mg, theta, epsilon, n_steps)

    # Production
    data_list = []
    prod_acc = 0
    
    for i in range(n_traj):
        phi, accepted, dH = hmc_step(phi, mg, theta, epsilon, n_steps)
        if accepted: prod_acc += 1
        
        sigma = chiral_condensate(phi)

        data_list.append({
            "run_label": label,
            "trajectory": i,
            "sigma": sigma,
            "accepted": accepted,
            "dH": dH,
            "Lx": Lx,
            "Lt": current_Lt,
            "g_over_T": current_fracT,
            "mass": mg,
            "theta": theta
        })
        
    df = pd.DataFrame(data_list)
    final_acc_rate = prod_acc / n_traj
    return df, final_acc_rate



