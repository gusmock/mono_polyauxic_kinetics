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

"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

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
    sum_phases = 0.0
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
    z0   = np.zeros(n_phases)
    r0   = np.ones(n_phases)
    lam0 = np.linspace(np.min(x), np.max(x), n_phases)

    theta0 = np.concatenate([[yi0, yf0], z0, r0, lam0])

    res = minimize(
        sse_loss, theta0, args=(x,y,n_phases,model),
        method="L-BFGS-B"
    )
    theta = res.x

    yi, yf = theta[0], theta[1]
    z   = theta[2:2+n_phases]
    r   = theta[2+n_phases:2+2*n_phases]
    lam = theta[2+2*n_phases:2+3*n_phases]

    p = np.exp(z-np.max(z))
    p = p/np.sum(p)

    yhat = polyauxic_function(x, yi, yf, p, r, lam, model)

    sse = np.sum((y-yhat)**2)
    sst = np.sum((y-np.mean(y))**2)
    R2  = 1 - sse/sst if sst>0 else 0.0
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
    res = y - yhat
    med = np.median(res)
    dev = np.abs(res - med)
    mad = np.median(dev)
    if mad < 1e-12:
        mad = 1e-12
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

    for _ in range(n_rep):
        U = np.random.rand(n_points)
        scales = dev_min + (dev_max-dev_min)*U
        noise = scales * np.random.normal(0,1,n_points)
        y_obs = y_gen + noise

        t_all.append(t_sim)
        y_all.append(y_obs)
        collected.append(y_obs)

    t_all = np.concatenate(t_all)
    y_all = np.concatenate(y_all)

    if use_rout:
        th_pre, _, yhat_pre = fit_polyauxic(t_all, y_all, model, n_phases)
        out = detect_outliers(y_all, yhat_pre)
        t_clean = t_all[~out]
        y_clean = y_all[~out]
        theta, metrics, _ = fit_polyauxic(t_clean, y_clean, model, n_phases)
    else:
        theta, metrics, _ = fit_polyauxic(t_all, y_all, model, n_phases)

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
        row[f"p{j+1}"]   = p_hat[j]
        row[f"r{j+1}"]   = rhat[j]
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

    # SIDEBAR
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

        n_rep   = st.number_input("Replicates", 1, 5, 3)
        n_points= st.number_input("Points per replicate", 5, 100, 20)
        n_tests = st.number_input("Monte Carlo tests", 1, 200, 20)

        use_rout = st.checkbox("Use ROUT removal?", value=True)

        run_btn = st.button("RUN MONTE CARLO")

    # MAIN
    st.title("Polyauxic Monte Carlo Simulator — Parallel Version")

    with st.expander("Instructions", expanded=True):
        st.markdown("""
        This tool performs Monte Carlo robustness analysis of polyauxic growth
        models (Boltzmann or Gompertz). Parameters are in the sidebar; results
        and plots appear here.

        Constraints:
        - yi < yf
        - p_j > 0 and sum(p_j) = 1
        - λ1 < λ2 < ... < λn
        """)

    # Validação
    if yi >= yf:
        st.error("yi must be < yf")
        st.stop()

    p_arr = np.array(p_list, float)
    if np.any(p_arr <= 0):
        st.error("All p_j must be > 0.")
        st.stop()
    p_arr = p_arr/np.sum(p_arr)

    lam_arr = np.array(lam_list, float)
    if not np.all(np.diff(lam_arr) > 0):
        st.error("λ must be strictly increasing.")
        st.stop()

    p_true = p_arr
    r_true = np.array(r_list, float)
    lam_true = lam_arr

    if run_btn:
        # Dominio
        t_sim = np.linspace(0, max(lam_true)*1.3, int(n_points))
        y_gen = polyauxic_function(t_sim, yi, yf, p_true, r_true, lam_true, model)

        # Margem vertical pequena, mas baseada só em yi/yf (não no ruído)
        dy = abs(yf - yi)
        y_min_plot = min(yi, yf) - 0.05*dy
        y_max_plot = max(yi, yf) + 0.05*dy

        cols = st.columns(2)

        with cols[0]:
            st.subheader("Generating function (noise-free)")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(t_sim, y_gen, 'k-', lw=2)
            ax.set_xlim(t_sim.min(), t_sim.max())
            ax.set_ylim(y_min_plot, y_max_plot)
            ax.set_xlabel("t")
            ax.set_ylabel("y")
            ax.grid(True, ls=":")
            plt.tight_layout()
            st.pyplot(fig)

        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()

        df, y_mean, y_std = monte_carlo_parallel(
            model, yi, yf, p_true, r_true, lam_true,
            dev_min, dev_max,
            n_rep, n_points, n_tests,
            t_sim, y_gen,
            use_rout, n_phases,
            progress_bar, status_text
        )

        with cols[1]:
            st.subheader("Mean ± std of all simulated data")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(t_sim, y_gen, 'k-', lw=2, label="Generating function")
            ax2.errorbar(
                t_sim, y_mean, yerr=y_std,
                fmt='o', color='red', ecolor='gray',
                capsize=3, label="MC mean ± std"
            )
            ax2.set_xlim(t_sim.min(), t_sim.max())
            ax2.set_ylim(y_min_plot, y_max_plot)
            ax2.set_xlabel("t")
            ax2.set_ylabel("y")
            ax2.grid(True, ls=":")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

        # Evolução dos parâmetros
        st.subheader("Parameter evolution across tests")
        figp, axes = plt.subplots(2, 2, figsize=(10,6))

        # yi,yf
        ax_00 = axes[0,0]
        ax_00.plot(df["test"], df["yi_hat"], 'o-', label="yi_hat")
        ax_00.plot(df["test"], df["yf_hat"], 'o-', label="yf_hat")
        ax_00.set_title("yi and yf")
        ax_00.set_xlabel("test")
        ax_00.grid(True, ls=":")
        ax_00.legend()

        # p_j
        ax_01 = axes[0,1]
        for j in range(n_phases):
            ax_01.plot(df["test"], df[f"p{j+1}"], 'o-', label=f"p{j+1}")
        ax_01.set_title("p_j")
        ax_01.set_xlabel("test")
        ax_01.grid(True, ls=":")
        ax_01.legend()

        # r_j
        ax_10 = axes[1,0]
        for j in range(n_phases):
            ax_10.plot(df["test"], df[f"r{j+1}"], 'o-', label=f"r{j+1}")
        ax_10.set_title("rmax_j")
        ax_10.set_xlabel("test")
        ax_10.grid(True, ls=":")
        ax_10.legend()

        # lambda_j
        ax_11 = axes[1,1]
        for j in range(n_phases):
            ax_11.plot(df["test"], df[f"lam{j+1}"], 'o-', label=f"λ{j+1}")
        ax_11.set_title("lambda_j")
        ax_11.set_xlabel("test")
        ax_11.grid(True, ls=":")
        ax_11.legend()

        plt.tight_layout()
        st.pyplot(figp)

        # Critérios de informação
        st.subheader("Information criteria (AIC, AICc, BIC)")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.plot(df["test"], df["AIC"], 'o-', label="AIC")
        ax3.plot(df["test"], df["AICc"], 'o-', label="AICc")
        ax3.plot(df["test"], df["BIC"], 'o-', label="BIC")
        ax3.set_xlabel("test")
        ax3.grid(True, ls=":")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)

        # R²
        st.subheader("R² and adjusted R²")
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.plot(df["test"], df["R2"], 'o-', label="R2")
        ax4.plot(df["test"], df["R2_adj"], 'o-', label="R2_adj")
        ax4.set_xlabel("test")
        ax4.grid(True, ls=":")
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)

        st.subheader("Results table")
        st.dataframe(df)


if __name__ == "__main__":
    main()
