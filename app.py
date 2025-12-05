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

This Streamlit application evaluates the robustness of polyauxic microbial 
growth kinetic models (Boltzmann or Gompertz) using a fully parallelized 
Monte Carlo simulation engine.

It is designed to:
1) Accept user-defined parameters for a polyauxic model with 1–10 phases.
2) Generate a "generating function" (the curve without noise).
3) Simulate multiple experimental replicates with random noise.
4) Fit each simulated dataset using:
       - preliminary fitting,
       - optional ROUT-like outlier detection (MAD-based),
       - final fitting only on clean points.
5) Extract fitted parameters, information criteria, R², adjusted R².
6) Perform the entire Monte Carlo procedure in parallel using Python’s 
   multiprocessing (ProcessPoolExecutor), fully protected by:
    
       if __name__ == "__main__":
           main()

   This ensures *safe parallel execution* inside Streamlit, avoiding reload 
   loops or zombie processes.

================================================================================
MATHEMATICAL MODEL
================================================================================
For n phases, the polyauxic model is:

   y(x) = y_i + (y_f - y_i) * Σ p_j * term_j(x)

Boltzmann term:
   term_j(x) = 1 / (1 + exp((4*rmax_j*(λ_j - x)) / ((y_f-y_i)*p_j) + 2))

Gompertz term:
   term_j(x) = exp( -exp( 1 + (rmax_j*e*(λ_j - x))/((y_f-y_i)*p_j) ) )

Where:
- y_i < y_f                                  (strict requirement)
- p_j > 0 and Σ p_j = 1                      (weights)
- λ_1 < λ_2 < ... < λ_n                      (phase ordering)
- rmax_j > 0                                 (growth rate parameters)

================================================================================
NOISE MODEL (Excel-compatible)
================================================================================
Each observed point is:

   y_obs = y_gen + ( dev_min + (dev_max-dev_min)*RAND() ) * NORMINV(RAND(),0,1)

Where RAND() is uniform(0,1) and NORMINV() is Gaussian noise.

================================================================================
FULL ALGORITHM
================================================================================

1. Build generating function y_gen(t_sim).
2. For each Monte Carlo test (parallelized):
      a) Generate each replicate (n_rep replicates).
      b) Add Excel-style noise.
      c) Concatenate time/response.
      d) PRE-FIT → ROUT (optional) → FINAL FIT.
      e) Save: yi_hat, yf_hat, p_hat_j, r_hat_j, λ_hat_j
         and SSE, R², R²_adj, AIC, AICc, BIC.
3. Collect all noisy y_obs for global statistics.
4. Compute global mean ± std of y_obs.
5. Produce plots:
      - Generating function vs global mean ± std
      - yi,yf evolution
      - p_j evolution
      - rmax_j evolution
      - λ_j evolution
      - Information criteria
      - R² and R²_adj

================================================================================
INSTRUCTIONS TO THE USER
================================================================================
1) Select model (Boltzmann or Gompertz).
2) Select number of phases (1–10).
3) Enter parameters ensuring:
       yi < yf
       p_j > 0 and sum = 1
       λ_1 < λ_2 < ... < λ_n
4) Set noise minimum and maximum.
5) Set number of replicates and number of points per replicate.
6) Set number of Monte Carlo tests.
7) Optionally enable ROUT removal.
8) Run the simulation.

All Monte Carlo tests are executed in parallel for maximum performance.

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
        sum_phases += p[j]*term
    return yi + (yf-yi)*sum_phases


# =============================================================================
# FITTING ENGINE + ROUT
# =============================================================================

def sse_loss(theta, x, y, n_phases, model):
    yi = theta[0]
    yf = theta[1]
    z  = theta[2:2+n_phases]
    r  = theta[2+n_phases:2+2*n_phases]
    lam= theta[2+2*n_phases:2+3*n_phases]

    p = np.exp(z-np.max(z))
    p = p/np.sum(p)

    yhat = polyauxic_function(x, yi, yf, p, r, lam, model)
    return np.sum((y-yhat)**2)

def fit_polyauxic(x, y, model, n_phases):
    yi0, yf0 = np.min(y), np.max(y)
    z0   = np.zeros(n_phases)
    r0   = np.ones(n_phases)
    lam0 = np.linspace(np.min(x), np.max(x), n_phases)

    theta0 = np.concatenate([[yi0, yf0], z0, r0, lam0])

    res = minimize(
        sse_loss, theta0, args=(x,y,n_phases,model),
        method="L-BFGS-B"
    )
    theta = res.x

    yi = theta[0]
    yf = theta[1]
    z  = theta[2:2+n_phases]
    r  = theta[2+n_phases:2+2*n_phases]
    lam= theta[2+2*n_phases:2+3*n_phases]

    p = np.exp(z-np.max(z))
    p = p/np.sum(p)

    yhat = polyauxic_function(x, yi, yf, p, r, lam, model)

    sse = np.sum((y-yhat)**2)
    sst = np.sum((y-np.mean(y))**2)
    R2  = 1 - sse/sst if sst>0 else 0
    n = len(y)
    k = len(theta)
    R2adj = 1 - (1-R2)*(n-1)/(n-k-1) if n > k+1 else np.nan

    AIC  = n*np.log(sse/n) + 2*k
    BIC  = n*np.log(sse/n) + k*np.log(n)
    AICc = AIC + (2*k*(k+1))/(n-k-1) if (n-k-1)>0 else np.inf

    metrics = dict(SSE=sse, R2=R2, R2_adj=R2adj,
                   AIC=AIC, AICc=AICc, BIC=BIC)

    return theta, metrics, yhat

def detect_outliers(y, yhat):
    res = y-yhat
    med = np.median(res)
    absdev = np.abs(res-med)
    mad = np.median(absdev)
    if mad < 1e-12:
        mad = 1e-12
    z = absdev/(1.4826*mad)
    return z>2.5


# =============================================================================
# PARALLEL MONTE CARLO ENGINE
# =============================================================================

def run_single_test(args):
    (test_id, model, yi, yf, p_true, r_true, lam_true,
     dev_min, dev_max, n_rep, n_points, t_sim, y_gen,
     use_rout, n_phases) = args

    collected_y = []

    # ----- Generate replicates -----
    t_all = []
    y_all = []

    for rep in range(n_rep):
        U = np.random.rand(n_points)
        scales = dev_min + (dev_max-dev_min)*U
        noise  = scales * np.random.normal(0,1,n_points)
        y_obs  = y_gen + noise

        t_all.append(t_sim)
        y_all.append(y_obs)
        collected_y.append(y_obs)

    t_all = np.concatenate(t_all)
    y_all = np.concatenate(y_all)

    # ----- Fit with optional ROUT -----
    if use_rout:
        theta_pre, _, yhat_pre = fit_polyauxic(t_all, y_all, model, n_phases)
        out = detect_outliers(y_all, yhat_pre)
        t_clean = t_all[~out]
        y_clean = y_all[~out]
        theta, metrics, yhat = fit_polyauxic(t_clean, y_clean, model, n_phases)
    else:
        theta, metrics, yhat = fit_polyauxic(t_all, y_all, model, n_phases)

    yi_hat = theta[0]
    yf_hat = theta[1]
    z_hat  = theta[2:2+n_phases]
    p_hat  = np.exp(z_hat-np.max(z_hat))
    p_hat  = p_hat/np.sum(p_hat)

    r_hat  = theta[2+n_phases:2+2*n_phases]
    lam_hat= theta[2+2*n_phases:2+3*n_phases]

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
        row[f"r{j+1}"] = r_hat[j]
        row[f"lam{j+1}"] = lam_hat[j]

    collected = np.vstack(collected_y)

    return row, collected


def monte_carlo_parallel(model, yi, yf, p_true, r_true, lam_true,
                         dev_min, dev_max, n_rep, n_points, n_tests,
                         t_sim, y_gen, use_rout, n_phases):

    tasklist = []
    for k in range(1, n_tests+1):
        tasklist.append((
            k, model, yi, yf, p_true, r_true, lam_true,
            dev_min, dev_max, n_rep, n_points, t_sim, y_gen,
            use_rout, n_phases
        ))

    results = []
    all_y = []

    with ProcessPoolExecutor() as ex:
        for row, block in ex.map(run_single_test, tasklist):
            results.append(row)
            all_y.append(block)

    df = pd.DataFrame(results)
    all_y_arr = np.vstack(all_y)

    y_mean = np.mean(all_y_arr, axis=0)
    y_std  = np.std(all_y_arr, axis=0)

    return df, y_mean, y_std


# =============================================================================
# STREAMLIT INTERFACE
# =============================================================================

def main():

    st.title("Polyauxic Monte Carlo Simulation — Parallel Version")

    with st.expander("Instructions", expanded=True):
        st.markdown("""
        **How to use this application:**

        1. Select model (Boltzmann or Gompertz).
        2. Choose number of phases (1–10).
        3. Enter parameters obeying constraints:
           - `yi < yf`
           - each `p_j > 0` and sum of all `p_j = 1`
           - `lambda_1 < lambda_2 < ... < lambda_n`
        4. Choose minimum and maximum absolute noise deviation.
        5. Choose:
             - number of replicates per test,
             - number of points per replicate,
             - number of Monte Carlo tests.
        6. Enable/Disable ROUT outlier removal.
        7. Run the simulation.
        """)

    st.header("Model setup")

    model = st.selectbox("Model", ["Boltzmann", "Gompertz"])
    n_phases = st.number_input("Number of phases", 1, 10, 2)

    yi = st.number_input("Initial value yi", value=0.0)
    yf = st.number_input("Final value yf", value=1.0)

    st.subheader("Phase parameters")
    p_list = []
    r_list = []
    lam_list = []

    for j in range(n_phases):
        p_list.append(
            st.number_input(f"p{j+1} (raw, >0)", value=1.0/n_phases, min_value=1e-9)
        )
        r_list.append(
            st.number_input(f"rmax{j+1}", value=0.1)
        )
        lam_list.append(
            st.number_input(f"lambda{j+1}", value=float(j+1))
        )

    # ----- PARAMETER VALIDATION -----
    # yi < yf
    if yi >= yf:
        st.error("Constraint violated: yi must be strictly less than yf.")
        st.stop()

    # p_j > 0 and sum=1
    p_arr = np.array(p_list, float)
    if np.any(p_arr <= 0):
        st.error("Constraint violated: all p_j must be > 0.")
        st.stop()
    p_arr = p_arr / np.sum(p_arr)

    # lambda strictly increasing
    lam_arr = np.array(lam_list, float)
    if not np.all(np.diff(lam_arr) > 0):
        st.error("Constraint violated: lambdas must satisfy λ1 < λ2 < ... < λn.")
        st.stop()

    p_true = p_arr
    r_true = np.array(r_list, float)
    lam_true = lam_arr

    # ----- NOISE AND SIMULATION SETTINGS -----

    st.header("Noise and Monte Carlo settings")
    dev_min = st.number_input("Absolute deviation (min)", value=0.0)
    dev_max = st.number_input("Absolute deviation (max)", value=0.1)

    n_rep = st.number_input("Replicates per test", 1, 5, 3)
    n_points = st.number_input("Points per replicate", 5, 100, 20)
    n_tests = st.number_input("Monte Carlo tests", 1, 100, 20)

    use_rout = st.checkbox("Use ROUT outlier removal?", value=True)

    if st.button("Run Simulation"):
        st.header("Generating function")
        t_sim = np.linspace(0, max(lam_true)*1.3, int(n_points))
        y_gen = polyauxic_function(t_sim, yi, yf, p_true, r_true, lam_true, model)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(t_sim, y_gen, 'k-', lw=2)
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_ylim(min(yi,yf)-0.05*abs(yf-yi), max(yi,yf)+0.05*abs(yf-yi))
        ax.grid(True, ls=":")
        st.pyplot(fig)

        st.header("Monte Carlo simulation (parallelized)")

        df, y_mean, y_std = monte_carlo_parallel(
            model, yi, yf, p_true, r_true, lam_true,
            dev_min, dev_max, n_rep, n_points, n_tests,
            t_sim, y_gen, use_rout, n_phases
        )

        st.success("Simulation completed.")

        # ----- PLOT: generating function vs Monte Carlo mean ± std -----
        st.header("Data distribution vs generating function")

        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(t_sim, y_gen, 'k-', lw=2, label="Generating function")
        ax2.errorbar(t_sim, y_mean, yerr=y_std, fmt='o', color='red',
                     ecolor='gray', label="Mean ± std (MC)")
        ax2.legend()
        ax2.grid(True, ls=":")
        ax2.set_xlabel("t")
        ax2.set_ylabel("y")
        st.pyplot(fig2)

        # ----- PARAMETER EVOLUTION PLOTS -----
        st.header("Parameter stability across Monte Carlo tests")

        figp, axes = plt.subplots(1,4,figsize=(18,4))

        # yi,yf
        axes[0].plot(df["test"], df["yi_hat"], 'o-', label="yi_hat")
        axes[0].plot(df["test"], df["yf_hat"], 'o-', label="yf_hat")
        axes[0].legend()
        axes[0].set_title("yi, yf")

        # p_j
        for j in range(n_phases):
            axes[1].plot(df["test"], df[f"p{j+1}"], 'o-', label=f"p{j+1}")
        axes[1].legend()
        axes[1].set_title("p_j")

        # r_j
        for j in range(n_phases):
            axes[2].plot(df["test"], df[f"r{j+1}"], 'o-', label=f"r{j+1}")
        axes[2].legend()
        axes[2].set_title("rmax_j")

        # lam_j
        for j in range(n_phases):
            axes[3].plot(df["test"], df[f"lam{j+1}"], 'o-', label=f"λ{j+1}")
        axes[3].legend()
        axes[3].set_title("lambda_j")

        for ax in axes:
            ax.grid(True, ls=":")
            ax.set_xlabel("test")
        st.pyplot(figp)

        # ----- INFORMATION CRITERIA -----
        st.header("Information criteria")

        figc, axc = plt.subplots(figsize=(7,4))
        axc.plot(df["test"], df["AIC"], 'o-', label="AIC")
        axc.plot(df["test"], df["AICc"], 'o-', label="AICc")
        axc.plot(df["test"], df["BIC"], 'o-', label="BIC")
        axc.legend()
        axc.grid(True, ls=":")
        axc.set_xlabel("test")
        st.pyplot(figc)

        # ----- R² and R² adj -----
        st.header("R² and adjusted R²")

        figr, axr = plt.subplots(figsize=(7,4))
        axr.plot(df["test"], df["R2"], 'o-', label="R2")
        axr.plot(df["test"], df["R2_adj"], 'o-', label="R2_adj")
        axr.legend()
        axr.grid(True, ls=":")
        axr.set_xlabel("test")
        st.pyplot(figr)

        st.header("Raw results table")
        st.dataframe(df)

# =============================================================================
# SAFE ENTRY POINT FOR PROCESSPOOL EXECUTOR
# =============================================================================
if __name__ == "__main__":
    main()
