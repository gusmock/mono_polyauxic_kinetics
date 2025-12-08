"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 
================================================================================
POLYAUXIC ROBUSTNESS SIMULATOR (v3.2 - ROUT & Visualization)
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)
GitHub: https://github.com/gusmock/mono_polyauxic_kinetics/

DESCRIPTION:
This Streamlit application performs rigorous Monte Carlo robustness testing for 
Polyauxic Kinetic Models. It incorporates the ROUT method (Robust regression 
and Outlier removal) based on Motulsky & Brown (2006) to handle outliers 
scientifically using False Discovery Rate (FDR).

REFERENCES:
1. Motulsky, H. J., & Brown, R. E. (2006). Detecting outliers when fitting 
   data with nonlinear regression – a new method based on robust nonlinear 
   regression and the false discovery rate. BMC Bioinformatics, 7, 123.
2. Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute 
   Deviation. Journal of the American Statistical Association.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress minor warnings for cleaner output during simulation
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. ROUT-like Outlier Detection (MAD-based)
# ------------------------------------------------------------
def detect_outliers(y_true, y_pred):
    """
    Detects outliers based on Median Absolute Deviation (MAD).
    Uses a modified Z-score.
    Returns a boolean mask where True indicates an outlier.
    """
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    # 1.4826 scales MAD to be consistent with standard deviation for normal distribution
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5 # Threshold Q (usually corresponds to FDR < 1% depending on N)

# ------------------------------------------------------------
# 1. Model Equations
# ------------------------------------------------------------
def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    # Avoid division by zero and overflow
    expo = (4 * r_j * (lam_j - t)) / (dy * p) + 2
    expo = np.clip(expo, -500, 500) 
    return p / (1 + np.exp(expo))

def gompertz_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    expo = (r_j * np.e * (lam_j - t)) / (dy * p) + 1
    expo = np.clip(expo, -500, 500)
    return p * np.exp(-np.exp(expo))

# ------------------------------------------------------------
# 2. Polyauxic generating function
# ------------------------------------------------------------
def polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    """Returns the sum and individual terms for plotting."""
    p_vec = np.asarray(p_vec, dtype=float)
    # Forced normalization
    if np.sum(p_vec) <= 0:
        p_vec = np.ones_like(p_vec) / len(p_vec)
    else:
        p_vec = p_vec / np.sum(p_vec)
    
    components = []
    sum_terms = np.zeros_like(t, dtype=float)
    
    for j in range(len(p_vec)):
        term = func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
        components.append((y_f - y_i) * term) # Scaled component
        sum_terms += term
        
    y_total = y_i + (y_f - y_i) * sum_terms
    return y_total, components, p_vec

def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    y, _, _ = polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func)
    return y

# ------------------------------------------------------------
# 3. Fitting & Helpers (Optimized)
# ------------------------------------------------------------
def polyauxic_fit_model(t, theta, func, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r = theta[2 + n_phases : 2 + 2*n_phases]
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    p = expz / np.sum(expz)
    sum_terms = np.zeros_like(t)
    for j in range(n_phases):
        sum_terms += func(t, y_i, y_f, p[j], r[j], lam[j])
    return y_i + (y_f - y_i) * sum_terms

def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    if np.any(y_pred < -0.1*np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    dy_s = np.convolve(dy, np.ones(5)/5, mode='same') if len(dy)>=5 else dy
    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guesses = []
    if len(peaks) > 0:
        idx = np.argsort(props['peak_heights'])[::-1][:n_phases]
        best = peaks[idx]
        for p in best: guesses.append((t[p], abs(dy_s[p])))
    while len(guesses) < n_phases:
        tspan = t.max() - t.min() if t.max() > t.min() else 1
        guesses.append((t.min() + tspan*(len(guesses)+1)/(n_phases+1), (y.max()-y.min())/(tspan/n_phases)))
    guesses = sorted(guesses, key=lambda x: x[0])
    theta0 = np.zeros(2 + 3*n_phases)
    theta0[0] = y.min(); theta0[1] = y.max()
    for i,(lam,r) in enumerate(guesses):
        theta0[2 + n_phases + i] = r
        theta0[2 + 2*n_phases + i] = lam
    return theta0

def fit_polyauxic(t_all, y_all, func, n_phases, fast_mode=False):
    """
    Fits the model.
    fast_mode=True reduces optimization precision for speed in Monte Carlo.
    """
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1
    t_n = t_all / t_scale
    y_n = y_all / y_scale
    
    theta0 = smart_guess(t_all, y_all, n_phases)
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale; th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0 
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases]*(t_scale/y_scale)
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases]/t_scale
    
    bounds = [(-0.2,1.5),(0,2)] + [(-10,10)]*n_phases + [(0,500)]*n_phases + [(-0.1,1.2)]*n_phases
    
    # Optimization settings based on mode
    if fast_mode:
        popsize = 5
        maxiter = 100
        mutation = (0.5, 1)
        recombination = 0.7
    else:
        popsize = 20
        maxiter = 800
        mutation = (0.5, 1)
        recombination = 0.7

    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))
    
    # Global Search
    res_de = differential_evolution(sse_loss, bounds, args=(t_n,y_n,func,n_phases),
                                    maxiter=maxiter, popsize=popsize, init=init_pop, 
                                    strategy="best1bin", mutation=mutation, recombination=recombination,
                                    polish=False, tol=1e-3)
    
    # Local Polish
    res = minimize(sse_loss, res_de.x, args=(t_n,y_n,func,n_phases), method="L-BFGS-B", bounds=bounds, tol=1e-8)
    
    th_n = res.x
    th = np.zeros_like(th_n)
    th[0]=th_n[0]*y_scale; th[1]=th_n[1]*y_scale
    th[2:2+n_phases]=th_n[2:2+n_phases]
    th[2+n_phases:2+2*n_phases]=th_n[2+n_phases:2+2*n_phases]*(y_scale/t_scale)
    th[2+2*n_phases:2+3*n_phases]=th_n[2+2*n_phases:2+3*n_phases]*t_scale
    
    y_pred = polyauxic_fit_model(t_all, th, func, n_phases)
    sse = np.sum((y_all - y_pred)**2)
    sst = np.sum((y_all - np.mean(y_all))**2)
    r2 = 1 - sse/sst if sst>0 else np.nan
    n = len(y_all); k = len(th)
    r2adj = 1 - (1-r2)*(n-1)/(n-k-1) if (n-k-1)>0 else np.nan
    aic = n*np.log(sse/n) + 2*k
    bic = n*np.log(sse/n) + k*np.log(n)
    aicc = aic + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf
    
    return th, {"SSE":sse,"R2":r2,"R2_adj":r2adj,"AIC":aic,"AICc":aicc,"BIC":bic}

# ------------------------------------------------------------
# 4. Monte Carlo Engine (Updated with Iterative ROUT & Error Handling)
# ------------------------------------------------------------
def monte_carlo_single(test_idx, func, ygen, t_sim, p_true, r_true, lam_true,
                       dev_min, dev_max, n_rep, n_points, y_i, y_f, use_rout):
    try:
        n_phases = len(p_true)
        t_all_list = []; y_all_list = []
        y_matrix = np.zeros((n_rep, n_points))
        
        # Generate noisy data
        for rep in range(n_rep):
            scales = dev_min + (dev_max-dev_min)*np.random.rand(n_points)
            noise = scales * np.random.normal(0,1,n_points)
            y_obs = ygen + noise
            t_all_list.append(t_sim); y_all_list.append(y_obs)
            y_matrix[rep, :] = y_obs
        t_all = np.concatenate(t_all_list); y_all = np.concatenate(y_all_list)
        
        # --- Fit Logic ---
        th = None
        met = None
        
        if use_rout:
            # Iterative ROUT implementation
            mask = np.ones(len(y_all), dtype=bool) # Start keeping all
            max_rout_iter = 5
            
            for iteration in range(max_rout_iter):
                t_curr = t_all[mask]
                y_curr = y_all[mask]
                
                # Fit on current 'clean' data (using fast_mode)
                if len(y_curr) < (2 + 3*n_phases) + 2:
                    break # Not enough points left
                    
                th_temp, met_temp = fit_polyauxic(t_curr, y_curr, func, n_phases, fast_mode=True)
                
                # Predict on CURRENT subset to check residuals relative to THIS fit
                y_pred_temp = polyauxic_fit_model(t_curr, th_temp, func, n_phases)
                
                # Detect outliers in the current subset
                outliers_in_subset = detect_outliers(y_curr, y_pred_temp)
                
                if not np.any(outliers_in_subset):
                    # No new outliers found, converged
                    th, met = th_temp, met_temp
                    break
                else:
                    # Map subset outliers back to global mask
                    # (This is tricky, simplified approach: remove from mask)
                    # We iterate through the current valid indices and set outliers to False
                    current_indices = np.where(mask)[0]
                    indices_to_remove = current_indices[outliers_in_subset]
                    mask[indices_to_remove] = False
                    
                    # Update final result pointer
                    th, met = th_temp, met_temp
            
            # Fallback if loop finishes without strict convergence (take last result)
            if th is None:
                th, met = fit_polyauxic(t_all, y_all, func, n_phases, fast_mode=True)
                
        else:
            # Standard fit without outlier removal
            th, met = fit_polyauxic(t_all, y_all, func, n_phases, fast_mode=True)
            
        # --- Parameter Extraction ---
        yi_hat = th[0]; yf_hat = th[1]
        z = th[2:2+n_phases]
        r = th[2+n_phases:2+2*n_phases]
        lam = th[2+2*n_phases:2+3*n_phases]
        
        z_shift = z - np.max(z)
        p_hat = np.exp(z_shift)/np.sum(np.exp(z_shift))
        
        row = {
            "test":test_idx, "yi_hat":yi_hat, "yf_hat":yf_hat,
            "SSE":met["SSE"],"R2":met["R2"],"R2_adj":met["R2_adj"],
            "AIC":met["AIC"],"AICc":met["AICc"],"BIC":met["BIC"]
        }
        for j in range(n_phases):
            row[f"p{j+1}"]=p_hat[j]; row[f"r{j+1}"]=r[j]; row[f"lam{j+1}"]=lam[j]
            
        return row, y_matrix

    except Exception as e:
        # Error handling to prevent crash
        row = {"test":test_idx, "yi_hat":np.nan, "yf_hat":np.nan, "SSE":np.nan, "R2":np.nan}
        return row, np.zeros((n_rep, n_points))

def monte_carlo(func, ygen, t_sim, p_true, r_true, lam_true,
                dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, use_rout):
    results = []; all_y_blocks = []
    progress = st.progress(0.0)
    status_text = st.empty()
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(monte_carlo_single, i+1, func, ygen, t_sim, p_true, r_true, lam_true,
                                   dev_min, dev_max, n_rep, n_points, y_i, y_f, use_rout): i+1 for i in range(n_tests)}
        done = 0
        for fut in as_completed(futures):
            row, y_mat = fut.result()
            # Filter out failed runs (NaNs)
            if not np.isnan(row.get("yi_hat", np.nan)):
                results.append(row)
                all_y_blocks.append(y_mat)
            done += 1
            progress.progress(done / n_tests)
            status_text.text(f"Running Monte Carlo: {done}/{n_tests}")
            
    status_text.text("Simulation finished.")
    
    if len(results) == 0:
        return pd.DataFrame(), None, None
        
    df = pd.DataFrame(results).sort_values("test")
    if len(all_y_blocks) > 0:
        all_y = np.vstack(all_y_blocks)
        return df, np.mean(all_y, axis=0), np.std(all_y, axis=0)
    else:
        return df, np.zeros_like(t_sim), np.zeros_like(t_sim)

# ------------------------------------------------------------
# 5. Streamlit App Interface
# ------------------------------------------------------------

st.title("Polyauxic Robustness Simulator")

# --- SIDEBAR INPUTS ---
model = st.sidebar.selectbox("Model",["Boltzmann (Eq 31)","Gompertz (Eq 32)"])
func = boltzmann_term if "Boltzmann" in model else gompertz_term

n_phases = st.sidebar.number_input("Number of phases",1,10,2)

st.sidebar.subheader("Global Parameters")
y_i = st.sidebar.number_input("y_i (Start)", value=0.0)

# Constraint 1: y_f must be > y_i. Using dynamic min_value.
y_f_min = y_i + 0.01
y_f = st.sidebar.number_input("y_f (End)", min_value=y_f_min, value=max(1.0, y_f_min))

p_inputs=[]; r_true=[]; lam_true=[]
st.sidebar.subheader("Phase Parameters")

last_lam = -0.1 

for j in range(n_phases):
    with st.sidebar.expander(f"Phase {j+1}", expanded=True):
        p_in = st.number_input(f"Raw Proportion (p{j+1})", min_value=0.01, value=1.0, key=f"p_in_{j}")
        r = st.number_input(f"Rate (r_max{j+1})", min_value=0.01, value=1.0, key=f"r_{j}")
        
        lam_min = last_lam + 0.1
        lam = st.number_input(f"Lag Time (λ{j+1})", min_value=lam_min, value=max(float(j+1), lam_min), key=f"lam_{j}")
        last_lam = lam 
        
        p_inputs.append(p_in)
        r_true.append(r)
        lam_true.append(lam)

total_p = sum(p_inputs)
p_true = [p / total_p for p in p_inputs]

st.sidebar.markdown("---")
st.sidebar.subheader("Noise & Simulation")
dev_min = st.sidebar.number_input("Noise Min (Abs)", min_value=0.0, value=0.0)
dev_max = st.sidebar.number_input("Noise Max (Abs)", min_value=0.0, value=0.1)
n_rep = st.sidebar.number_input("Replicates", 1, 10, 3)
n_points = st.sidebar.number_input("Points/Rep", 10, 500, 50)
n_tests = st.sidebar.number_input("MC Tests", 1, 500, 20)
use_rout = st.sidebar.checkbox("Use ROUT Outlier Removal?", value=False)

# ------------------------------------------------------------
# LIVE PREVIEW
# ------------------------------------------------------------
st.subheader("True Parameters Preview")

max_lam = max(lam_true)
tmax = max(3*max_lam, 1.0)
t_sim = np.linspace(0, tmax, n_points)

ygen, components, p_norm_vec = polyauxic_components(t_sim, y_i, y_f, p_true, r_true, lam_true, func)

col_g1, col_g2 = st.columns([2, 1])

with col_g1:
    fig_prev, ax_prev = plt.subplots(figsize=(6, 4))
    ax_prev.plot(t_sim, ygen, 'k-', lw=2.5, label="Total Curve")
    colors = plt.cm.viridis(np.linspace(0, 1, n_phases))
    for j, (comp_y, color) in enumerate(zip(components, colors)):
        ax_prev.plot(t_sim, y_i + comp_y, ls='--', lw=1.5, color=color, label=f"Phase {j+1}")
        
    ax_prev.set_title(f"Generating Function ({model})")
    ax_prev.set_xlabel("Time")
    ax_prev.set_ylabel("Response (y)")
    ax_prev.grid(True, ls=':', alpha=0.6)
    ax_prev.legend(fontsize='small')
    st.pyplot(fig_prev)
    plt.close(fig_prev) # Memory cleanup

with col_g2:
    st.markdown("**Effective Parameters:**")
    st.write(f"**y_i:** {y_i:.2f} | **y_f:** {y_f:.2f}")
    df_params = pd.DataFrame({
        "Phase": [f"#{j+1}" for j in range(n_phases)],
        "p (norm)": [f"{v:.3f}" for v in p_norm_vec],
        "r_max": r_true,
        "lambda": lam_true
    })
    st.table(df_params)
    
    if abs(total_p - 1.0) > 1e-6:
        st.info(f"Note: Input p sums to {total_p:.2f}. Values normalized.")

# ------------------------------------------------------------
# MONTE CARLO EXECUTION
# ------------------------------------------------------------
st.markdown("---")
run = st.button("Run Monte Carlo Simulation", type="primary")

if run:
    df, y_mean, y_std = monte_carlo(
        func, ygen, t_sim, p_true, r_true, lam_true,
        dev_min, dev_max, n_rep, n_points, n_tests, y_i, y_f, use_rout
    )

    if not df.empty and y_mean is not None:
        dy = abs(y_f - y_i) if abs(y_f - y_i) > 0 else 1.0
        y_min_plot = min(y_i, y_f) - 0.1*dy
        y_max_plot = max(y_i, y_f) + 0.1*dy

        st.subheader("Simulation Results")
        
        # Aggregated Results
        st.markdown("### Global Mean ± Std (Simulated)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_sim, ygen, 'k--', lw=2, label="True Curve")
        ax.errorbar(t_sim, y_mean, yerr=y_std, fmt='o', color='blue', 
                    ecolor='lightblue', alpha=0.6, capsize=3, markersize=4, label="Sim Mean ± Std")
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.grid(True, ls=':')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) # Cleanup

        st.dataframe(df.head(10))
        st.download_button("Download Full CSV", df.to_csv(index=False), "monte_carlo_results.csv", "text/csv")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Information Criteria")
            fig, ax = plt.subplots(figsize=(6, 4))
            if "AIC" in df.columns:
                for col in ["AIC", "BIC"]:
                    ax.plot(df["test"], df[col], marker='o', ms=4, label=col)
                ax.legend()
            ax.grid(True, ls=':')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("#### Fit Quality (R²)")
            fig, ax = plt.subplots(figsize=(6, 4))
            if "R2_adj" in df.columns:
                ax.plot(df["test"], df["R2_adj"], color='green', marker='s', ms=4, label="Adj R²")
                ax.legend()
            ax.grid(True, ls=':')
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("#### Parameter Recovery Distribution")
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        axs[0,0].boxplot([df["yi_hat"], df["yf_hat"]], labels=["yi", "yf"])
        axs[0,0].set_title("Bounds Parameters")
        
        axs[0,1].boxplot([df[f"p{j+1}"] for j in range(n_phases)], labels=[f"p{j+1}" for j in range(n_phases)])
        axs[0,1].set_title("Proportions (p)")
        
        axs[1,0].boxplot([df[f"r{j+1}"] for j in range(n_phases)], labels=[f"r{j+1}" for j in range(n_phases)])
        axs[1,0].set_title("Rates (r_max)")
        
        axs[1,1].boxplot([df[f"lam{j+1}"] for j in range(n_phases)], labels=[f"λ{j+1}" for j in range(n_phases)])
        axs[1,1].set_title("Lags (λ)")
        
        for ax in axs.flat: ax.grid(True, ls=':', alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.error("Simulation failed to produce valid results. Check parameters or try increasing replicates.")
# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")

profile_pic_url = "https://github.com/gusmock.png" 

footer_html = f"""
<style>
    .footer-container {{
        width: 100%;
        font-family: sans-serif;
        color: #444;
        margin-bottom: 20px;
    }}
    
    /* Layout Flex para Foto + Texto */
    .profile-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 15px;
    }}
    
    /* Estilo da Foto */
    .profile-img {{
        width: 80px;
        height: 80px;
        border-radius: 50%;       /* Deixa redonda */
        object-fit: cover;
        border: 2px solid #e0e0e0; /* Borda sutil */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .profile-info h4 {{
        margin: 0;
        font-size: 1.1rem;
        color: #222;
    }}
    
    .profile-info p {{
        margin: 2px 0 0 0;
        font-size: 0.9rem;
        color: #666;
    }}

    .badge-container {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
    }}
    
    .badge-container img {{
        height: 28px;
    }}
</style>

<div class="footer-container">
    
    <div class="profile-header">
        <img src="{profile_pic_url}" class="profile-img" alt="Gustavo Mockaitis">
        <div class="profile-info">
            <h2>Development: Prof. Dr. Gustavo Mockaitis</h2>
            <h4>GBMA / FEAGRi / UNICAMP</h4>
            <p>Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and Environment, School of Agricultural Engineering, University of Campinas (GBMA/FEAGRI/UNICAMP), 397 Michel Debrun Street, CEP 13.083-875, Campinas, SP, Brazil.</p>
        </div>
    </div>

    <div class="badge-container">
        <a href="https://arxiv.org/abs/2507.05960" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2507.05960-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
        </a>
        <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank">
            <img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID">
        </a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=pt-BR&oi=ao" target="_blank">
            <img src="https://img.shields.io/badge/Scholar-Perfil-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar">
        </a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/ResearchGate-Perfil-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate">
        </a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank">
            <img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes">
        </a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Conectar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank">
            <img src="https://img.shields.io/badge/Web_of_Science-Perfil-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science">
        </a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/UNICAMP-Institucional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP">
        </a>
    </div>
</div>
"""

st_components.html(footer_html, height=280, scrolling=False)
