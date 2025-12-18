import streamlit as st
import pandas as pd
import numpy as np
import polyauxic_lib as lib
import io
import time

# ==============================================================================
# CONFIGURATION & UI
# ==============================================================================
st.set_page_config(page_title="Batch Experiment Runner", layout="wide")

st.title("🧪 Polyauxic Experimental Design Runner")
st.markdown("""
This app runs the **Full Factorial Design** experiment on uploaded datasets.
It evaluates:
* **Models:** Gompertz vs. Boltzmann
* **Constraints:** Floating vs. Forced ($y_i=0$)
* **Robustness:** ROUT FDR (0.5%, 1.0%, 1.5%)
* **Parsimony:** Threshold ($\Delta > 0$ vs. $\Delta > 2$)
""")

# --- Sidebar Configuration ---
st.sidebar.header("Experimental Factors")

# Factor A: Models
st.sidebar.subheader("Factor A: Models")
use_gompertz = st.sidebar.checkbox("Gompertz", value=True)
use_boltzmann = st.sidebar.checkbox("Boltzmann", value=True)

# Factor B: Constraints
st.sidebar.subheader("Factor B: Constraints")
use_floating = st.sidebar.checkbox("Floating y_i", value=True)
use_forced = st.sidebar.checkbox("Forced y_i=0", value=True)

# Factor C: FDR
st.sidebar.subheader("Factor C: ROUT FDR (%)")
fdr_levels = st.sidebar.multiselect("Select Levels", [0.5, 1.0, 1.5], default=[0.5, 1.0, 1.5])

# Factor D: Thresholds
st.sidebar.subheader("Factor D: Thresholds")
thresh_levels = st.sidebar.multiselect("Select Delta IC", [0.0, 2.0], default=[0.0, 2.0])

# --- File Uploader ---
st.header("1. Upload Datasets")
uploaded_files = st.file_uploader(
    "Upload your CSV files (must contain keywords: '1storder', 'replicates', 'unfinished' in filename)", 
    accept_multiple_files=True,
    type=["csv", "xlsx"]
)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} files.")
    
    # Preview classes
    file_map = []
    for f in uploaded_files:
        fname = f.name.lower()
        if "replicates" in fname: c = "Replicates"
        elif "1storder" in fname: c = "First_Order"
        elif "unfinished" in fname: c = "Unfinished"
        else: c = "Unknown (Will Skip)"
        file_map.append({"Filename": f.name, "Detected Class": c})
    
    st.dataframe(pd.DataFrame(file_map), height=200)

    # --- Execution Logic ---
    if st.button("RUN BATCH EXPERIMENT", type="primary"):
        if not (use_gompertz or use_boltzmann):
            st.error("Please select at least one Model.")
            st.stop()
            
        # Build Factor Lists based on UI
        models_to_run = []
        if use_gompertz: models_to_run.append(("Gompertz", lib.gompertz_term_eq32))
        if use_boltzmann: models_to_run.append(("Boltzmann", lib.boltzmann_term_eq31))
        
        constraints_to_run = []
        if use_floating: constraints_to_run.append(False)
        if use_forced: constraints_to_run.append(True)
        
        if not constraints_to_run or not fdr_levels or not thresh_levels:
            st.error("Please ensure at least one option is selected for all factors.")
            st.stop()

        # Progress Bar
        total_steps = len(uploaded_files) * len(models_to_run) * len(constraints_to_run) * len(fdr_levels)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        step_count = 0
        
        start_time = time.time()

        for file_obj in uploaded_files:
            filename = file_obj.name
            
            # Identify Class
            lower_name = filename.lower()
            if "replicates" in lower_name: class_type = "Replicates"
            elif "1storder" in lower_name: class_type = "First_Order"
            elif "unfinished" in lower_name: class_type = "Unfinished"
            else: continue # Skip unknown
            
            # Read Data
            try:
                if filename.endswith(".xlsx"): df = pd.read_excel(file_obj)
                else: df = pd.read_csv(file_obj)
                t_flat, y_flat, _ = lib.process_data(df)
            except:
                continue

            # Nested Loops ( Factors A, B, C )
            for model_name, model_func in models_to_run:
                for force_yi in constraints_to_run:
                    for fdr in fdr_levels:
                        
                        # Update UI
                        step_count += 1
                        progress = min(step_count / total_steps, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {filename} | {model_name} | Yi={'Forced' if force_yi else 'Float'} | FDR={fdr}")

                        # --- CORE LOGIC (Copied from run_experiment.py) ---
                        phase_fits = []
                        
                        # Fit phases 1 to 5
                        for n in range(1, 6):
                            # Pre-fit for Outliers
                            res_robust = lib.fit_model_auto_robust_pre(t_flat, y_flat, model_func, n, force_yi, False)
                            
                            if res_robust:
                                mask = lib.detect_outliers_rout_rigorous(y_flat, res_robust['y_pred'], Q=fdr)
                                t_c = t_flat[~mask] if np.any(mask) else t_flat
                                y_c = y_flat[~mask] if np.any(mask) else y_flat
                                
                                # Final Fit
                                phase_fits.append(lib.fit_model_auto(t_c, y_c, model_func, n, force_yi, False))
                            else:
                                phase_fits.append(None)
                                
                        valid_fits = [f for f in phase_fits if f is not None]
                        if not valid_fits: continue
                        
                        aicc_values = [res['metrics']['AICc'] for res in valid_fits]

                        # Factor D (Threshold) - Applied post-hoc
                        for thresh in thresh_levels:
                            best_idx = lib.select_first_local_min_index(aicc_values, threshold=thresh)
                            winner = valid_fits[best_idx]
                            
                            # Extract SE Rate
                            n_ph = winner['n_phases']
                            start_r = 2 + n_ph
                            end_r = 2 + 2 * n_ph
                            se_rates = winner['se'][start_r : end_r]
                            avg_se = np.nanmean(se_rates) if not np.all(np.isnan(se_rates)) else np.nan

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
                                "Avg_SE_Rate": avg_se,
                                "Outliers": np.sum(winner['outliers'])
                            })

        elapsed = time.time() - start_time
        st.success(f"Experiment Complete in {elapsed:.1f} seconds!")
        
        # Results Display & Download
        if all_results:
            res_df = pd.DataFrame(all_results)
            st.dataframe(res_df)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Final Results CSV",
                csv,
                "experiment_results.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("No results generated.")
