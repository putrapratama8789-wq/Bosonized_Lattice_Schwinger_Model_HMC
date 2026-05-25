import numpy as np
import pandas as pd
import os
import math
import multiprocessing
from functools import partial
from hmc_core import *
from config import *
from utils import *


# WORKER FUNCTION (Loops over choosed variable, 
# change depend on what variable you want to loop over)

def solve_point(val_theta, fixed_params):

    val_T = fixed_params['val_T']
    LT_REF = fixed_params['LT_REF']
    FIXED_MASS = fixed_params['FIXED_MASS']

    
    # Calculate Lt for the fixed Temperature
    target_Lt = int(math.ceil(val_T / ag))
    if target_Lt < 2: target_Lt = 1 # Safety floor

    theta = val_theta #looping over theta (the current variable)

    
    # Auto-Tune Epsilon
    try:
        opt_epsilon = auto_tune_epsilon(target_Lt, FIXED_MASS, theta, n_steps=N_STEPS)
    except:
        opt_epsilon = 0.1 # Fallback
    
    #  Run Simulation 
    label = f"m_{val_theta:.2f}" # Updated label
    df_target, acc_target = run_simulation(target_Lt, FIXED_MASS, val_theta, N_THERM, N_TRAJ, opt_epsilon, N_STEPS, label=label)
    
    # Save Raw Data
    FOLDER_RAW = "Massive Schwinger Model/raw"
    save_custom_raw(df_target, FOLDER_RAW, FIXED_MASS, theta, label="target_only")
    
    # Analysis
    mean_T, err_T = jackknife_error(df_target['sigma'].values, bin_size=BIN_SIZE)
    Sigma_Renormalized = mean_T
    Delta_Sigma = abs(Sigma_Renormalized)
    
    print(f"--> DONE m={FIXED_MASS:.2f} | Acc={acc_target:.1%} | |Delta|={Delta_Sigma:.5f}", flush=True)
    
    # Summary Dictionary
    summary_result = {
        "mass": FIXED_MASS,           
        "g": g,
        "ag": ag,
        "theta": theta,
        "fracT_target": val_T,   
        "Lx": Lx,
        "Lt_target": target_Lt,
        "Lt_ref_used": LT_REF, 
        "Delta_Sigma": Delta_Sigma,      
        "Error": err_T,
        "Sigma_Raw_Target": mean_T,
        "Error_Target": err_T,
        "n_therm": N_THERM,
        "n_traj": N_TRAJ,
        "n_steps": N_STEPS,
        "acc_target": acc_target,
        "epsilon": opt_epsilon,
        "jackknife_bin_size": BIN_SIZE,
    }

    # Save individual Summary
    FOLDER_SUMMARY = "Massive Schwinger Model/summary"
    save_custom_summary(summary_result, FOLDER_SUMMARY, FIXED_MASS, theta)

    return summary_result



# MAIN EXECUTION 
if __name__ == "__main__":
        
    print(f"ag = {ag} | Fixed T = {TEMP_REF} | g = {g} | Lx = {Lx}")

    FOLDER_FULL = "Massive Schwinger Model/full_summary"
    
    print(f"=== Starting Multicore Simulation ===")
    n_jobs = multiprocessing.cpu_count()
    print(f"CPU Cores Available: {n_jobs}")
    print(f"Masses to simulate: {FIXED_MASS}")
    
    # Fixed Parameters
    fixed_params = {
        'val_T': TEMP_REF,
        'LT_REF': LT_REF,
        'FIXED_MASS': FIXED_MASS,
    }

    worker_func = partial(solve_point, fixed_params=fixed_params)
    
    print(f"\nLaunching {n_jobs} parallel workers...")
    
    all_runs_summary = []
    
    with multiprocessing.Pool(processes=n_jobs) as pool:
        
        results = pool.map(worker_func, theta_input) #input the variabel value to the worker function
        all_runs_summary.extend(results)

    # FULL MASTER SUMMARY
    print("\n" + "="*50)
    print("ALL TASKS COMPLETE. Saving Master Summary...")
    
    if not os.path.exists(FOLDER_FULL):
        os.makedirs(FOLDER_FULL)
        
    df_full = pd.DataFrame(all_runs_summary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    full_summary_name = f"{FOLDER_FULL}/full_summary_T{TEMP_REF:.2f}_ALL_MASSES_{timestamp}.csv"
    
    df_full.to_csv(full_summary_name, index=False)
    
    print(f"Full summary saved to: {full_summary_name}")
    print("="*50)

