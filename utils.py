import numpy as np
from hmc_core import *
from config import *
import os
import pandas as pd
from scipy.integrate import quad
from scipy.special import ellipk



def jackknife_error(data_array, bin_size):
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



def auto_tune_epsilon(current_Lt, current_mg, current_theta, start_epsilon=0.01, n_steps=20, n_test_traj=400, target_acc=0.70, tolerance=0.05):

    print(f"\n--- Auto-Tuning Epsilon (Target: {target_acc:.0%}, Tol: {tolerance:.0%}) ---")
    print(f"Tuning on Lattice: {Lx} x {current_Lt}") 
    
    # test range for epsilon
    test_range = np.arange(start_epsilon, 1, 0.005)
    
    # Initialize a dummy field
    phi = np.random.randn(Lx, current_Lt) 
    
    #thermalize 
    print("Pre-thermalizing for tuner...")
    safe_eps = 0.05
    for _ in range(200):
        phi, _, _ = hmc_step(phi, current_mg, current_theta, safe_eps, n_steps)

    best_eps = start_epsilon
    min_diff = 1.0


    print(f"{'Epsilon':<10} | {'Acceptance':<12} | {'Status':<10}")
    print("-" * 40)
    
    for eps in test_range:
        accepted_count = 0
        
        # Fast Test Loop
        for _ in range(n_test_traj):
            phi, accepted, _ = hmc_step(phi, current_mg, current_theta, eps, n_steps)
            if accepted: accepted_count += 1
            
        rate = accepted_count / n_test_traj
        diff = abs(rate - target_acc)
        
        # Check the new best
        is_best = False
        if diff < min_diff:
            min_diff = diff
            best_eps = eps
            is_best = True
        
        status = "*" if is_best else ""
        print(f"{eps:<10.3f} | {rate:<12.1%} | {status}")
        
        # EARLY EXIT
        if diff < tolerance:
            print(f">> Early Match Found! Stopping search.")
            break
            

        if rate < (target_acc - 0.20): 
            print(f">> Acceptance dropping too low. Stopping.")
            break

    print("-" * 40)
    print(f">> Optimal Epsilon found: {best_eps:.3f}")
    return best_eps


# Saving Funtions
def save_custom_raw(dataframe, folder, mass, theta, label):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{folder}/raw_m{mass:.2f}_T{theta:.2f}_a{a}_{label}.csv"
    dataframe.to_csv(filename, index=False)
    print(f"[Saved] Raw data: {filename}")

def save_custom_summary(summary_dict, folder, mass, theta):
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{folder}/summary_m{mass:.2f}_T{theta:.2f}_a{a}.csv"
    
    df = pd.DataFrame([summary_dict])
    df.to_csv(filename, index=False)
    print(f"[Saved] Run Summary: {filename}")




def sigma_free_lattice(m, ag):
    x = 1.0/(ag**2)
    k = 1.0 / (1.0 + (m**2)/(g**2*x))

    K_value = ellipk(k)

    free_dirac = (m / (np.pi * g)) * (1.0 / np.sqrt(1.0 + (m**2)/(g**2*x))) * K_value

    if m == 0.0:
        free_dirac = 0.0

    return free_dirac

def sigma_free_finiteN(m, g, a, N):
    # using Eq. (3.1) discrete sum
    x = 1.0/(g**2 * a**2)
    mu = 2.0 * m / (g**2 * a)   
    q = np.arange(1, N//2 + 1)
    denom = np.sqrt(mu**2 + 4.0*x**2*(np.cos(q*np.pi/(N+1))**2))
    sigma = (np.sqrt(x)/N) * np.sum(mu/denom)
    return sigma  

def sigma_free_euclidean(m, g, ag, Lx, Lt):

    a = ag / g
    ma = m * a
    
    nx = np.arange(Lx)
    ntau = np.arange(Lt)

    kx = 2.0 * np.pi * nx / Lx
    ktau = np.pi * (2.0 * ntau + 1.0) / Lt
    
    KX, KTAU = np.meshgrid(kx, ktau, indexing='ij')
    
    denominator = np.sin(KX)**2 + np.sin(KTAU)**2 + ma**2
    
    dimensionless_sum = np.mean(ma / denominator)

    free_condensate = - (1.0 / a) * dimensionless_sum
    
    return free_condensate



if __name__ == "__main__":
    
    m_test = 0.5
    a_test = 0.025
    T_test = 1.0 / (a_test * 448)  # reference temperature for Ltau=448

    Sigma_free = sigma_free_lattice(m=m_test, ag=a_test)
    c = np.exp(gamma)/(2.0*(np.pi)**(3/2)) * UV_val
    phys_sigma_free = Sigma_free * UV_val
    print(f"Free Dirac Chiral Condensate at T={T_test:.5f}, m={m_test}, a={a_test}: Sigma_free = {Sigma_free:.5f}")
    print(f"Physical Sigma_free (with prefactor): {phys_sigma_free:.5f}")
    print(f"Banuls sigma_free (finite N=896): {sigma_free_finiteN(m=m_test, g=g, a=a_test, N=896):.5f}")
    print(f"Banuls sigma_free (euclidean): {sigma_free_euclidean(m=m_test, g=g, ag=a_test, Lx=896, Lt=448):.5f}")
    opt_epsilon = auto_tune_epsilon(112, FIXED_MASS, current_theta=0.0, n_steps=N_STEPS)
    