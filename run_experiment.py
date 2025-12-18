import pandas as pd
import numpy as np
import os
import polyauxic_lib as lib 

# ================= CONFIGURATION =================
DATASET_FOLDER = "datasets/"  # Place your 15 CSV files here
OUTPUT_FILE = "final_experiment_results.csv"

# EXPERIMENTAL FACTORS
MODELS = [
    ("Gompertz", lib.gompertz_term_eq32), 
    ("Boltzmann", lib.boltzmann_term_eq31)
]
CONSTRAINTS = [False, True]  # False = Floating yi, True = Forced yi=0
FDR_LEVELS = [0.5, 1.0, 1.5] # ROUT Q values
THRESHOLDS = [0.0, 2.0]      # Factor D: Strict vs Conservative IC selection

# DATASET MAP (Example - User must update this)
# Format: "filename.csv": "Class_Category"
DATASET_MAP = {
    # Class 1: First-order-like
    "dataset_01.csv": "First_Order",
    "dataset_02.csv": "First_Order",
    "dataset_03.csv": "First_Order",
    "dataset_04.csv": "First_Order",
    "dataset_05.csv": "First_Order",
    
    # Class 2: Replicates
    "dataset_06.csv": "Replicates",
    "dataset_07.csv": "Replicates",
    "dataset_08.csv": "Replicates",
    "dataset_09.csv": "Replicates",
    "dataset_10.csv": "Replicates",
    
    # Class 3: Unfinished
    "dataset_11.csv": "Unfinished",
    "dataset_12.csv": "Unfinished",
    "dataset_13.csv": "Unfinished",
    "dataset_14.csv": "Unfinished",
    "dataset_15.csv": "Unfinished",
}

# ================= MAIN LOOP =================
def run_batch():
    all_results = []
    
    if not os.path.exists(DATASET_FOLDER):
        print(f"Error: Folder '{DATASET_FOLDER}' not found. Please create it and add your CSV files.")
        return

    print("Starting Experimental Run...")
    print(f"Total Conditions per Dataset: {len(MODELS) * len(CONSTRAINTS) * len(FDR_LEVELS) * len(THRESHOLDS)}")
    
    # Loop 1: Datasets (Block)
    for filename, class_type in DATASET_MAP.items():
        filepath = os.path.join(DATASET_FOLDER, filename)
        
        if not os.path.exists(filepath):
            print(f"  [Skipped] File not found: {filename}")
            continue
            
        print(f"  Processing {filename} ({class_type})...")
        
        try:
            # Load Data Once
            if filename.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            
            t_flat, y_flat, _ = lib.process_data(df)
            
            if len(t_flat) == 0:
                print(f"    Error: No data found in {filename}")
                continue

        except Exception as e:
            print(f"    Error reading {filename}: {e}")
            continue

        # Loop 2: Model (Factor A)
        for model_name, model_func in MODELS:
            
            # Loop 3: Constraint (Factor B)
            for force_yi in CONSTRAINTS:
                
                # Loop 4: FDR (Factor C)
                for fdr in FDR_LEVELS:
                    
                    # PERFORMANCE OPTIMIZATION:
                    # We fit phases 1 to 5 ONCE for this combination of (Model, Constraint, FDR).
                    # Then we apply the different Thresholds (Factor D) to the *results* of these fits.
                    
                    phase_fits = [] # Stores fit results for n=1, n=2... n=5
                    
                    # Fit phases 1 to 5
                    for n in range(1, 6):
                        # 1. Detect Outliers (ROUT)
                        # Pre-fit to get residuals
                        res_robust = lib.fit_model_auto_robust_pre(
                            t_flat, y_flat, model_func, n, 
                            force_yi=force_yi, force_yf=False
                        )
                        
                        if res_robust:
                            mask = lib.detect_outliers_rout_rigorous(
                                y_flat, res_robust['y_pred'], Q=fdr
                            )
                            
                            # Apply Mask if outliers found
                            if np.any(mask):
                                t_clean = t_flat[~mask]
                                y_clean = y_flat[~mask]
                            else:
                                t_clean, y_clean = t_flat, y_flat
                                
                            # 2. Final Fit (SSE)
                            fit_res = lib.fit_model_auto(
                                t_clean, y_clean, model_func, n,
                                force_yi=force_yi, force_yf=False
                            )
                            phase_fits.append(fit_res)
                        else:
                            phase_fits.append(None)

                    # Filter out failed fits (None) for selection
                    valid_fits = [f for f in phase_fits if f is not None]
                    
                    if not valid_fits:
                        continue

                    # Extract AICc values for the valid fits
                    aicc_values = [res['metrics']['AICc'] for res in valid_fits]

                    # Loop 5: Threshold (Factor D)
                    # Now we select the best phase count based on the Threshold
                    for thresh in THRESHOLDS:
                        best_idx = lib.select_first_local_min_index(aicc_values, threshold=thresh)
                        winner = valid_fits[best_idx]
                        
                        # Calculate Mean SE for Rate (r_max) parameters
                        # Theta structure: [yi, yf, z...z, r...r, l...l]
                        # r_max starts at index: 2 + n_phases
                        n_ph = winner['n_phases']
                        start_r = 2 + n_ph
                        end_r = 2 + 2 * n_ph
                        se_rates = winner['se'][start_r : end_r]
                        
                        # Handle NaN in SE
                        if np.all(np.isnan(se_rates)):
                            avg_se_rate = np.nan
                        else:
                            avg_se_rate = np.nanmean(se_rates)

                        # Store Result Row
                        all_results.append({
                            "Dataset": filename,
                            "Class": class_type,
                            "Model": model_name,
                            "Constraint_Yi": "Forced=0" if force_yi else "Floating",
                            "FDR": fdr,
                            "Threshold": thresh,
                            "Selected_Phases": n_ph,
                            "AICc": winner['metrics']['AICc'],
                            "R2_Adj": winner['metrics']['R2_adj'],
                            "SSE": winner['metrics']['SSE'],
                            "Avg_SE_Rate": avg_se_rate,
                            # Save outliers count for verification
                            "Outliers_Detected": np.sum(winner['outliers'])
                        })

    # Save to CSV
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Experiment Complete! Results saved to: {OUTPUT_FILE}")
    else:
        print("\n❌ No results generated. Check your dataset folder.")

if __name__ == "__main__":
    run_batch()
