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
Polyauxic Robustness Simulator
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)

This Streamlit application evaluates robustness of polyauxic microbial growth
kinetic models (Boltzmann or Gompertz) via a fully parallel Monte Carlo engine.

It reproduces the structure of the original polyauxic modeling application:
    - PARAMETERS in sidebar (left)
    - RESULTS and GRAPHS in main page
    - instructions in collapsible box

FEATURES:
---------
1. Select model (Boltzmann or Gompertz).
2. Choose number of phases (1–10).
3. Enter parameters:
       yi < yf
       p_j > 0 and Σp_j = 1
       lambdas strictly increasing
4. Choose noise min/max amplitude.
5. Choose number of replicates per test.
6. Choose number of Monte Carlo tests.
7. ROUT outlier removal option (as in original).
8. Parallel execution with ProcessPoolExecutor.
9. Progress bar updated in real time.
10. Plots:
       - generating function
       - mean ± std of MC data
       - yi,yf evolution
       - p_j evolution
       - rmax_j evolution
       - λ_j evolution
       - AIC/AICc/BIC
       - R², R² adj
================================================================================
MATHEMATICAL MODEL
================================================================================

y(x) = y_i + (y_f - y_i) * Σ p_j * term_j(x)

Boltzmann term:
 term_j(x) = 1/(1+exp((4*rmax_j*(λ_j - x))/((y_f-y_i)*p_j) + 2))

Gompertz term:
 term_j(x) = exp(-exp(1 + (rmax_j*e*(λ_j-x))/((y_f-y_i)*p_j)))

REQUIREMENTS:
    yi < yf
    p_j > 0 and Σp_j = 1
    λ_1 < λ_2 < ... < λ_n

================================================================================
NOISE MODEL
================================================================================
Excel-compatible:
    y_obs = y_gen + ( dev_min + (dev_max-dev_min)*RAND() ) * NORMINV(RAND(),0,1)

================================================================================
FULL ALGORITHM
================================================================================

For each Monte Carlo test:
    1. Simulate replicates with noise.
    2. Preliminary fit.
    3. ROUT (MAD-based) outlier removal (optional).
    4. Final fit with clean data.
    5. Save parameters and statistics.

All tests run in parallel via ProcessPoolExecutor.

================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
import math


st.set_page_config(layout="wide")

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def boltzmann_term(x, yi, yf, p, rmax, lam):
    delta = yf - yi
    if delta == 0:
        delta = 1e-12
    exponent = (4*rmax*(lam-x))/(delta*p) + 2
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (1.0 + np.exp(exponent))


def gompertz_term(x, yi, yf, p, rmax, lam):
    delta = yf - yi
    if delta == 0:
        delta = 1e-12
    exponent = 1 + (rmax*np.e*(lam-x))/(delta*p)
    exponent = np.clip(exponent, -500, 500)
    return np.exp(-np.exp(exponent))


def polyauxic_function(x, yi, yf, p, rmax, lam, model):
    x = np.asarray(x)
    sum_phases = 0
    for j in range(len(p)):
        if model == "Boltzmann":
            term = boltzmann_term(x, yi, yf, p[j], rmax[j], lam[j])
        else:
            term = gompertz_term(x, yi, yf, p[j], rmax[j], lam[j])
        sum_phases += p[j] * term
    return yi + (yf - yi) * sum_phases


# =============================================================================
# FITTING ENGINE + ROUT
# =============================================================================

def sse_loss(theta, x, y, n_phases, model):
    yi = theta[0]
    yf = theta[1]
    z  = theta[2:2+n_phases]
    r  = theta[2+n_phases:2+2*n_phases]
    lam= theta[2+2*n_phases:2+3*n_phases]

    p = np.exp(z - np.max(z))
    p = p / np.sum(p)

    yhat = polyauxic_function(x, yi, yf, p, r, lam, model)
    return np.sum((y-yhat)**2)


def fit_polyauxic(x, y, model, n_phases):
    yi0, yf0 = np.min(y), np.max(y)
    z0 = np.zeros(n_phases)
    r0 = np.ones(n_phases)
    lam0 = np.linspace(np.min(x), np.max(x), n_phases)

    theta0 = np.concatenate([[yi0, yf0], z0, r0, lam0])

    res = minimize(
        sse_loss, theta0, args=(x,y,n_phases,model),
        method="L-BFGS-B"
    )
    theta = res.x

    yi, yf = theta[0], theta[1]
    z = theta[2:2+n_phases]
    r = theta[2+n_phases:2+2*n_phases]
    lam = theta[2+2*n_phases:2+3*n_phases]

    p = np.exp(z-np.max(z))
    p = p/np.sum(p)

    yhat = polyauxic_function(x, yi, yf, p, r, lam, model)

    sse = np.sum((y-yhat)**2)
    sst = np.sum((y-np.mean(y))**2)
    R2  = 1 - sse/sst if sst>0 else 0
    n = len(y)
    k = len(theta)
    R2adj = 1 - (1-R2)*(n-1)/(n-k-1) if n > k+1 else np.nan

    AIC = n*np.log(sse/n) + 2*k
    BIC = n*np.log(sse/n) + k*np.log(n)
    AICc = AIC + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf

    metrics = dict(SSE=sse, R2=R2, R2_adj=R2adj, AIC=AIC, AICc=AICc, BIC=BIC)

    return theta, metrics, yhat


def detect_outliers(y, yhat):
    res = y - yhat
    med = np.median(res)
    dev = np.abs(res - med)
    mad = np.median(dev)
    if mad < 1e-12: mad = 1e-12
    z = dev/(1.4826*mad)
    return z > 2.5


# =============================================================================
# PARALLEL MONTE CARLO ENGINE
# =============================================================================

def run_single_test(args):
    (test_id, model, yi, yf, p_true, r_true, lam_true,
     dev_min, dev_max, n_rep, n_points,
     t_sim, y_gen, use_rout, n_phases) = args

    collected = []

    t_all = []
    y_all = []

    # Simulate replicates
    for rep in range(n_rep):
        U = np.random.rand(n_points)
        scales = dev_min + (dev_max-dev_min)*U
        noise = scales * np.random.normal(0,1,n_points)
        y_obs = y_gen + noise

        t_all.append(t_sim)
        y_all.append(y_obs)
        collected.append(y_obs)

    t_all = np.concatenate(t_all)
    y_all = np.concatenate(y_all)

    # Fit with optional ROUT
    if use_rout:
        th_pre, _, yhat_pre = fit_polyauxic(t_all, y_all, model, n_phases)
        out = detect_outliers(y_all, yhat_pre)
        t_clean = t_all[~out]
        y_clean = y_all[~out]
        theta, metrics, yhat = fit_polyauxic(t_clean, y_clean, model, n_phases)
    else:
        theta, metrics, yhat = fit_polyauxic(t_all, y_all, model, n_phases)

    yi_hat, yf_hat = theta[0], theta[1]
    z = theta[2:2+n_phases]
    p_hat = np.exp(z-np.max(z))
    p_hat = p_hat/np.sum(p_hat)

    rhat = theta[2+n_phases:2+2*n_phases]
    lam_hat = theta[2+2*n_phases:2+3*n_phases]

    row = {
        "test": test_id,
        "yi_hat": yi_hat,
        "yf_hat": yf_hat,
        "SSE": metrics["SSE"],
        "R2": metrics["R2"],
        "R2_adj": metrics["R2_adj"],
        "AIC": metrics["AIC"],
        "AICc": metrics["AICc"],
        "BIC": metrics["BIC"]
    }

    for j in range(n_phases):
        row[f"p{j+1}"] = p_hat[j]
        row[f"r{j+1}"] = rhat[j]
        row[f"lam{j+1}"] = lam_hat[j]

    return row, np.vstack(collected)


def monte_carlo_parallel(model, yi, yf, p_true, r_true, lam_true,
                         dev_min, dev_max, n_rep, n_points, n_tests,
                         t_sim, y_gen, use_rout, n_phases,
                         progress_bar, status_text):

    tasks = []
    for k in range(1, n_tests+1):
        tasks.append((k, model, yi, yf, p_true, r_true, lam_true,
                      dev_min, dev_max, n_rep, n_points,
                      t_sim, y_gen, use_rout, n_phases))

    results = []
    all_y = []

    with ProcessPoolExecutor() as ex:
        futures = ex.map(run_single_test, tasks)

        for k, (row, block) in enumerate(futures):
            progress_bar.progress((k+1)/n_tests)
            status_text.text(f"Running test {k+1}/{n_tests}...")
            results.append(row)
            all_y.append(block)

    progress_bar.empty()
    status_text.text("Done.")

    df = pd.DataFrame(results)
    y_all_arr = np.vstack(all_y)

    y_mean = np.mean(y_all_arr, axis=0)
    y_std  = np.std(y_all_arr, axis=0)

    return df, y_mean, y_std


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("Polyauxic Parameters")

        model = st.selectbox("Model", ["Boltzmann", "Gompertz"])
        n_phases = st.number_input("Number of phases", 1, 10, 2)

        yi = st.number_input("yi", value=0.0)
        yf = st.number_input("yf", value=1.0)

        st.subheader("Phase parameters")

        p_list = []
        r_list = []
        lam_list = []

        for j in range(n_phases):
            p_list.append(
                st.number_input(f"p{j+1} (>0)", min_value=1e-9, value=1.0/n_phases)
            )
            r_list.append(
                st.number_input(f"rmax{j+1}", value=0.1)
            )
            lam_list.append(
                st.number_input(f"lambda{j+1}", value=float(j+1))
            )

        st.subheader("Noise and Monte Carlo")

        dev_min = st.number_input("Deviation (min)", value=0.0)
        dev_max = st.number_input("Deviation (max)", value=0.1)

        n_rep = st.number_input("Replicates", 1, 5, 3)
        n_points = st.number_input("Points per replicate", 5, 100, 20)
        n_tests = st.number_input("Monte Carlo tests", 1, 200, 20)

        use_rout = st.checkbox("Use ROUT removal?", value=True)

        run_btn = st.button("RUN MONTE CARLO")

    # ========== MAIN INTERFACE ==========
    st.title("Polyauxic Monte Carlo Simulator — Parallel Version")

    with st.expander("Instructions", expanded=True):
        st.markdown("""
        This tool performs Monte Carlo robustness analysis of polyauxic growth
        models, using the Boltzmann or Gompertz functions. Parameters are entered
        in the left sidebar; results and plots appear here.

        Steps:
        1. Choose your model and number of phases.
        2. Enter parameters yi, yf, p_j, r_j, λ_j.
        3. Set noise and Monte Carlo parameters.
        4. Click RUN.

        All Monte Carlo tests run in parallel using multiple CPU cores.
        """)

    # Validate constraints
    if yi >= yf:
        st.error("yi must be < yf")
        st.stop()

    p_arr = np.array(p_list, float)
    if np.any(p_arr <= 0):
        st.error("All p_j must be > 0.")
        st.stop()
    p_arr = p_arr/np.sum(p_arr)

    lam_arr = np.array(lam_list, float)
    if not np.all(np.diff(lam_arr)>0):
        st.error("λ must be strictly increasing.")
        st.stop()

    p_true = p_arr
    r_true = np.array(r_list, float)
    lam_true = lam_arr

    if run_btn:

        # GENERATING FUNCTION
        st.header("Generating Function")
        t_sim = np.linspace(0, max(lam_true)*1.3, int(n_points))
        y_gen = polyauxic_function(t_sim, yi, yf, p_true, r_true, lam_true, model)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(t_sim, y_gen, 'k-', lw=2)
        ax.set_ylim(min(yi,yf)-0.05*abs(yf-yi), max(yi,yf)+0.05*abs(yf-yi))
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.grid(True, ls=":")
        st.pyplot(fig)

        # PROGRESS BAR
        progress_bar = st.progress(0)
        status_text = st.empty()

        # MONTE CARLO PARALLEL
        df, y_mean, y_std = monte_carlo_parallel(
            model, yi, yf, p_true, r_true, lam_true,
            dev_min, dev_max,
            n_rep, n_points, n_tests,
            t_sim, y_gen,
            use_rout, n_phases,
            progress_bar, status_text
        )

        # MEAN ± STD
        st.header("Mean ± Standard Deviation of Monte Carlo Data")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(t_sim, y_gen, 'k-', lw=2, label="Generating Function")
        ax2.errorbar(t_sim, y_mean, yerr=y_std, fmt='o', color='red', 
                     ecolor='gray', label="MC Mean ± Std")
        ax2.grid(True, ls=":")
        ax2.legend()
        st.pyplot(fig2)

        # PARAMETER EVOLUTION
        st.header("Parameter Evolution")

        figp, axes = plt.subplots(1,4, figsize=(18,4))

        # yi,yf
        axes[0].plot(df["test"], df["yi_hat"], 'o-', label="yi_hat")
        axes[0].plot(df["test"], df["yf_hat"], 'o-', label="yf_hat")
        axes[0].legend()
        axes[0].set_title("yi and yf")
        axes[0].grid(True)

        # p_j
        for j in range(n_phases):
            axes[1].plot(df["test"], df[f"p{j+1}"], 'o-', label=f"p{j+1}")
        axes[1].legend()
        axes[1].set_title("p_j")
        axes[1].grid(True)

        # r_j
        for j in range(n_phases):
            axes[2].plot(df["test"], df[f"r{j+1}"], 'o-', label=f"r{j+1}")
        axes[2].legend()
        axes[2].set_title("rmax_j")
        axes[2].grid(True)

        # lambda_j
        for j in range(n_phases):
            axes[3].plot(df["test"], df[f"lam{j+1}"], 'o-', label=f"λ{j+1}")
        axes[3].legend()
        axes[3].set_title("lambda_j")
        axes[3].grid(True)

        st.pyplot(figp)

        # INFORMATION CRITERIA
        st.header("Information Criteria")
        fig3, ax3 = plt.subplots(figsize=(7,4))
        ax3.plot(df["test"], df["AIC"], 'o-', label="AIC")
        ax3.plot(df["test"], df["AICc"], 'o-', label="AICc")
        ax3.plot(df["test"], df["BIC"], 'o-', label="BIC")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

        # R2
        st.header("R² and Adjusted R²")
        fig4, ax4 = plt.subplots(figsize=(7,4))
        ax4.plot(df["test"], df["R2"], 'o-', label="R2")
        ax4.plot(df["test"], df["R2_adj"], 'o-', label="R2_adj")
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig4)

        # TABLE
        st.header("Results Table")
        st.dataframe(df)


# =============================================================================
# MAIN GUARD FOR PARALLEL EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
