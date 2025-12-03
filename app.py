"""
                                                                                 @@                      
                    ::++        ++..                                ..######  ########  @@@@                    
                    ++++      ..++++                                ##########  ########  @@@                    
                    ++++++    ++++++                                ..####  ########  ##########  ..##                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########  ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########@@########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######  ##########  ##            
            ++++++::####        ####++++++++                  ####  ########  ######  ######  ########  ####            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####++########  ########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++        ##########  ######  ####  ####  ######  ##########          
  --++++++++++##    ##          @@::  mm##++++++++++      ##############  ######MM##  ####MM####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ++  ::##########  ####  ##  ##  ####  ############            
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    ######--      
        ++++##..      MM  ..######--    ##::++++          ##########    @@####              ######    ############      
        ++++@@++    ####  ##########    ##++++++          ################                  @@################      
    ++++++++::##          ##########    ##++++++++++      ##################                  ####################  @@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                                #@@@@#
    mm++::++++++##  ##++              ##++++++++++mm        ################                  ####################  @@  
          ++++++####                ##::++++                ############--                    ##################        
            ++++++MM##@@        ####::++++++                  ######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                mm  ::########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@                          
                    ++++        ++++                                                    --..    @@@@                          



"""

"""
================================================================================
      .---.                  
    / .-. \                 &
   | (   ) |               &&&
    \ `-' /               &&&
      `---'              &&&
                        &   
     [UNICAMP]         [GBMA]
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
import io

# ==============================================================================
# 0. CONFIGURATION & TRANSLATIONS
# ==============================================================================

# Global Plot Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 12

LANGUAGES = {
    "English": "en",
    "Português (BR)": "pt",
    "Français (CA)": "fr"
}

TEXTS = {
    "title": {
        "en": "Polyauxic Modeling (Replicates & Information Criteria)",
        "pt": "Modelagem Poliauxica (Réplicas e Critérios de Informação)",
        "fr": "Modélisation Polyauxique (Réplicats et Critères d'Information)"
    },
    "sidebar_settings": {
        "en": "Settings",
        "pt": "Configurações",
        "fr": "Paramètres"
    },
    "response_type": {
        "en": "Response Type (Y Axis)",
        "pt": "Tipo de Resposta (Eixo Y)",
        "fr": "Type de Réponse (Axe Y)"
    },
    "upload_label": {
        "en": "Upload CSV/XLSX (Col pairs: t1, y1, t2, y2...)",
        "pt": "Arquivo CSV/XLSX (Pares colunas: t1, y1, t2, y2...)",
        "fr": "Télécharger CSV/XLSX (Paires col: t1, y1, t2, y2...)"
    },
    "max_phases": {
        "en": "Max Phases to Test",
        "pt": "Máximo de Fases para testar",
        "fr": "Phases Max à Tester"
    },
    "info_upload": {
        "en": "Load a file. Format: Col A=Time1, B=Resp1, C=Time2, D=Resp2, etc.",
        "pt": "Carregue um arquivo. Formato: Col A=Tempo1, B=Resp1, C=Tempo2, D=Resp2, etc.",
        "fr": "Chargez un fichier. Format: Col A=Temps1, B=Resp1, C=Temps2, D=Resp2, etc."
    },
    "data_loaded": {
        "en": "Data Loaded: {0} replicates identified. Total points: {1}",
        "pt": "Dados Carregados: {0} réplicas identificadas. Total de pontos: {1}",
        "fr": "Données Chargées: {0} réplicats identifiés. Points totaux: {1}"
    },
    "run_button": {
        "en": "RUN COMPARATIVE ANALYSIS",
        "pt": "EXECUTAR ANÁLISE COMPARATIVA",
        "fr": "LANCER L'ANALYSE COMPARATIVE"
    },
    "tab_gompertz": {"en": "Gompertz (Eq. 32)", "pt": "Gompertz (Eq. 32)", "fr": "Gompertz (Eq. 32)"},
    "tab_boltzmann": {"en": "Boltzmann (Eq. 31)", "pt": "Boltzmann (Eq. 31)", "fr": "Boltzmann (Eq. 31)"},
    "expanding": {
        "en": "{0}: Fitting {1} Phase(s)",
        "pt": "{0}: Ajuste com {1} Fase(s)",
        "fr": "{0}: Ajustement avec {1} Phase(s)"
    },
    "optimizing": {
        "en": "Optimizing {0} phases...",
        "pt": "Otimizando {0} fases...",
        "fr": "Optimisation de {0} phases..."
    },
    "warning_insufficient": {
        "en": "Insufficient data.",
        "pt": "Dados insuficientes.",
        "fr": "Données insuffisantes."
    },
    "table_title": {
        "en": "Model Selection Table",
        "pt": "Tabela de Seleção de Modelo",
        "fr": "Tableau de Sélection du Modèle"
    },
    "best_model_msg": {
        "en": "🏆 Best Suggested Model: **{0} Phase(s)** (Based on lowest AICc).",
        "pt": "🏆 Melhor Modelo Sugerido: **{0} Fase(s)** (Baseado no menor AICc).",
        "fr": "🏆 Meilleur Modèle Suggéré: **{0} Phase(s)** (Basé sur le plus bas AICc)."
    },
    "graph_summary_title": {
        "en": "Effect of Phase Count on Criteria",
        "pt": "Efeito do Número de Fases nos Critérios",
        "fr": "Effet du Nombre de Phases sur les Critères"
    },
    "download_plot": {
        "en": "Download Plot (SVG)",
        "pt": "Baixar Gráfico (SVG)",
        "fr": "Télécharger le Graphique (SVG)"
    },
    "download_summary": {
        "en": "Download Summary (SVG)",
        "pt": "Baixar Resumo (SVG)",
        "fr": "Télécharger le Résumé (SVG)"
    },
    "axis_time": {"en": "Time (h/d)", "pt": "Tempo (h/d)", "fr": "Temps (h/j)"},
    "legend_global": {"en": "Global Fit", "pt": "Ajuste Global", "fr": "Ajustement Global"},
    "legend_phase": {"en": "Phase {0}", "pt": "Fase {0}", "fr": "Phase {0}"},
    "legend_mean": {"en": "Mean (w/o Outliers)", "pt": "Média (s/ Outliers)", "fr": "Moyenne (sans Aberrants)"},
    "legend_outlier": {"en": "Outliers", "pt": "Outliers", "fr": "Valeurs Aberrantes"},
    "error_read": {
        "en": "Error processing data: {0}",
        "pt": "Erro ao processar dados: {0}",
        "fr": "Erreur de traitement: {0}"
    },
    "error_cols": {
        "en": "Column error.",
        "pt": "Erro nas colunas.",
        "fr": "Erreur de colonne."
    }
}

# Dictionary for variable labels
VAR_LABELS = {
    "Genérico y(t)": {
        "en": ("Response (y)", ("y_i", "y_f"), "r_max"),
        "pt": ("Resposta (y)", ("y_i", "y_f"), "r_max"),
        "fr": ("Réponse (y)", ("y_i", "y_f"), "r_max")
    },
    "Produto P(t)": {
        "en": ("Product Conc. (P)", ("P_i", "P_f"), "r_P,max"),
        "pt": ("Concentração de Produto (P)", ("P_i", "P_f"), "r_P,max"),
        "fr": ("Concentration en Produit (P)", ("P_i", "P_f"), "r_P,max")
    },
    "Substrato S(t)": {
        "en": ("Substrate Conc. (S)", ("S_i", "S_f"), "r_S,max"),
        "pt": ("Concentração de Substrato (S)", ("S_i", "S_f"), "r_S,max"),
        "fr": ("Concentration en Substrat (S)", ("S_i", "S_f"), "r_S,max")
    },
    "Biomassa X(t)": {
        "en": ("Biomass Conc. (X)", ("X_i", "X_f"), "µ_max"),
        "pt": ("Concentração Celular (X)", ("X_i", "X_f"), "µ_max"),
        "fr": ("Concentration Cellulaire (X)", ("X_i", "X_f"), "µ_max")
    }
}

# ==============================================================================
# 1. MATHEMATICAL MODELS
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Phase term j for Boltzmann model (Eq. 31)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Phase term j for Gompertz model (Eq. 32)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """Global Model: Weighted sum of sigmoidal phases."""
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    
    z = theta[2 : 2+n_phases]
    r_max = theta[2+n_phases : 2+2*n_phases]
    lambda_ = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])

    return y_i + (y_f - y_i) * sum_phases

# ==============================================================================
# 2. LOSS FUNCTIONS & STATISTICS
# ==============================================================================

def sse_loss(theta, t, y, model_func, n_phases):
    """Objective Function: Sum of Squared Errors (SSE)."""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): # Light physical penalty
        return 1e12
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Numerical Hessian for error estimation."""
    k = len(theta)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i = np.zeros(k); e_i[i] = epsilon
            e_j = np.zeros(k); e_j[j] = epsilon
            f_pp = func(theta + e_i + e_j, *args)
            f_pm = func(theta + e_i - e_j, *args)
            f_mp = func(theta - e_i + e_j, *args)
            f_mm = func(theta - e_i - e_j, *args)
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
    return hess

def detect_outliers(y_true, y_pred):
    """Visual method for marking outliers (ROUT-based)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def smart_initial_guess(t, y, n_phases):
    """Heuristic to find initial r_max and lambda."""
    dy = np.gradient(y, t)
    dy_smooth = np.convolve(dy, np.ones(5)/5, mode='same')
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth)*0.1, distance=min_dist)
    
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lambda': t[p_idx], 'r_max': dy_smooth[p_idx]})
            
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lambda': t.min() + t_span * (len(guesses)+1)/(n_phases+1),
            'r_max': (np.max(y)-np.min(y)) / (t_span/n_phases)
        })
    guesses.sort(key=lambda x: x['lambda'])
    
    theta_guess = np.zeros(2 + 3*n_phases)
    theta_guess[0] = np.min(y)
    theta_guess[1] = np.max(y)
    theta_guess[2:2+n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2+n_phases+i] = guesses[i]['r_max']
        theta_guess[2+2*n_phases+i] = guesses[i]['lambda']
    return theta_guess

# ==============================================================================
# 3. FITTING ENGINE
# ==============================================================================

def calculate_p_errors(z_vals, cov_z):
    """Calculates standard error for p (Softmax) via Delta Method."""
    exps = np.exp(z_vals - np.max(z_vals))
    p = exps / np.sum(exps)
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: J[i, j] = p[i] * (1 - p[i])
            else: J[i, j] = -p[i] * p[j]
    try:
        cov_p = J @ cov_z @ J.T
        se_p = np.sqrt(np.abs(np.diag(cov_p)))
        return se_p
    except:
        return np.full(n, np.nan)

def fit_model_auto(t_data, y_data, model_func, n_phases):
    
    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params:
        return None 

    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    theta_smart = smart_initial_guess(t_data, y_data, n_phases)
    theta0_norm = np.zeros_like(theta_smart)
    theta0_norm[0] = theta_smart[0] / y_scale
    theta0_norm[1] = theta_smart[1] / y_scale
    theta0_norm[2:2+n_phases] = 0.0
    theta0_norm[2+n_phases:2+2*n_phases] = theta_smart[2+n_phases:2+2*n_phases] / (y_scale/t_scale)
    theta0_norm[2+2*n_phases:2+3*n_phases] = theta_smart[2+2*n_phases:2+3*n_phases] / t_scale
    
    pop_size = 50 
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    bounds = []
    bounds.append((-0.2, 1.5)) 
    bounds.append((0.0, 2.0))
    for _ in range(n_phases): bounds.append((-10, 10))
    for _ in range(n_phases): bounds.append((0.0, 500.0))
    for _ in range(n_phases): bounds.append((-0.1, 1.2))

    res_de = differential_evolution(
        sse_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
        seed=None, polish=True, tol=1e-6
    )
    
    res_opt = minimize(
        sse_loss, res_de.x, args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B', bounds=bounds, tol=1e-10
    )
    
    theta_norm = res_opt.x
    
    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)
    se_p = np.full(n_phases, np.nan)

    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2:2+n_phases] = theta_norm[2:2+n_phases]
    scale_r = y_scale / t_scale
    theta_real[2+n_phases:2+2*n_phases] = theta_norm[2+n_phases:2+2*n_phases] * scale_r
    scale_l = t_scale
    theta_real[2+2*n_phases:2+3*n_phases] = theta_norm[2+2*n_phases:2+3*n_phases] * scale_l
    
    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm)**2)
        
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2:2+n_phases] = se_norm[2:2+n_phases] 
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * scale_r
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * scale_l
        
        idx_z_start = 2
        idx_z_end = 2 + n_phases
        cov_z = cov_norm[idx_z_start:idx_z_end, idx_z_start:idx_z_end]
        z_vals = theta_norm[idx_z_start:idx_z_end]
        se_p = calculate_p_errors(z_vals, cov_z)
        
    except:
        se_real = np.full_like(theta_real, np.nan)

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)
    
    sse = np.sum((y_data - y_pred)**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - sse/sst
    
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12
    
    if (n_len - k - 1) > 0:
        r2_adj = 1 - (1 - r2) * (n_len - 1) / (n_len - k - 1)
    else:
        r2_adj = np.nan

    aic = n_len * np.log(sse/n_len) + 2*k
    bic = n_len * np.log(sse/n_len) + k * np.log(n_len)
    aicc = aic + (2*k*(k+1))/(n_len-k-1) if (n_len-k-1)>0 else np.inf
    
    return {
        "n_phases": n_phases,
        "theta": theta_real,
        "se": se_real,
        "se_p": se_p,
        "metrics": {"R2": r2, "R2_adj": r2_adj, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. DATA PROCESSING
# ==============================================================================

def process_data(df):
    """Processes DataFrame detecting replicates in pairs of columns."""
    df = df.dropna(axis=1, how='all')
    cols = df.columns.tolist()
    
    all_t = []
    all_y = []
    replicates = []

    num_replicates = len(cols) // 2
    
    for i in range(num_replicates):
        t_col = cols[2*i]
        y_col = cols[2*i+1]
        
        t_vals = pd.to_numeric(df[t_col], errors='coerce').values
        y_vals = pd.to_numeric(df[y_col], errors='coerce').values
        
        mask = ~np.isnan(t_vals) & ~np.isnan(y_vals)
        t_clean = t_vals[mask]
        y_clean = y_vals[mask]
        
        all_t.extend(t_clean)
        all_y.extend(y_clean)
        replicates.append({'t': t_clean, 'y': y_clean, 'name': f'Replica {i+1}'})
        
    t_flat = np.array(all_t)
    y_flat = np.array(all_y)
    idx_sort = np.argsort(t_flat)
    
    return t_flat[idx_sort], y_flat[idx_sort], replicates

def calculate_mean_with_outliers(replicates, model_func, theta, n_phases):
    """Calculates mean excluding outliers based on the model fit."""
    all_data = []
    for rep in replicates:
        for t, y in zip(rep['t'], rep['y']):
            all_data.append({'t': t, 'y': y})
    df_all = pd.DataFrame(all_data)
    
    y_pred_all = polyauxic_model(df_all['t'].values, theta, model_func, n_phases)
    outliers_mask = detect_outliers(df_all['y'].values, y_pred_all)
    df_all['is_outlier'] = outliers_mask
    df_all['t_round'] = df_all['t'].round(4) 
    
    grouped = df_all[~df_all['is_outlier']].groupby('t_round')['y'].agg(['mean', 'std']).reset_index()
    return grouped, df_all

# ==============================================================================
# 5. INTERFACE & VISUALIZATION
# ==============================================================================

def plot_metrics_summary(results_list, model_name, lang):
    """Generates a summary chart of metrics vs phases."""
    phases = [r['n_phases'] for r in results_list]
    aic = [r['metrics']['AIC'] for r in results_list]
    aicc = [r['metrics']['AICc'] for r in results_list]
    bic = [r['metrics']['BIC'] for r in results_list]
    r2_adj = [r['metrics']['R2_adj'] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Criteria
    ax1.plot(phases, aic, 'o--', label='AIC')
    ax1.plot(phases, aicc, 's-', label='AICc')
    ax1.plot(phases, bic, '^:', label='BIC')
    ax1.set_xlabel('Number of Phases')
    ax1.set_ylabel('Information Criteria Value')
    ax1.set_title('Information Criteria vs Phases')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot R2
    ax2.plot(phases, r2_adj, 'o-', color='purple', label='Adjusted R²')
    ax2.set_xlabel('Number of Phases')
    ax2.set_ylabel('Adjusted R²')
    ax2.set_title('Adjusted R² vs Phases')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    st.download_button(
        label=TEXTS['download_summary'][lang],
        data=buf.getvalue(),
        file_name=f"metrics_summary_{model_name}.svg", 
        mime="image/svg+xml",
        key=f"dl_summary_{model_name}" 
    )
    st.pyplot(fig)

def display_single_fit(res, replicates, model_func, color_main, y_label, param_labels, rate_label, lang):
    n = res['n_phases']
    theta = res['theta']
    se = res['se']
    se_p = res['se_p']
    
    yi_name, yf_name = param_labels
    
    stats_df, raw_data_w_outliers = calculate_mean_with_outliers(replicates, model_func, theta, n)
    
    y_i, y_f = theta[0], theta[1]
    y_i_se, y_f_se = se[0], se[1]
    
    z = theta[2:2+n]
    r_max = theta[2+n:2+2*n]
    r_max_se = se[2+n:2+2*n]
    lambda_ = theta[2+2*n:2+3*n]
    lambda_se = se[2+2*n:2+3*n]
    
    p = np.exp(z - np.max(z))
    p /= np.sum(p)
    
    phases = []
    for i in range(n):
        phases.append({
            "p": p[i], "SE p": se_p[i],
            "r_max": r_max[i], "r_max_se": r_max_se[i],
            "lambda": lambda_[i], "lambda_se": lambda_se[i]
        })
    phases.sort(key=lambda x: x['lambda'])
    
    c_plot, c_data = st.columns([1.5, 1])
    
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for rep in replicates:
            ax.scatter(rep['t'], rep['y'], color='gray', alpha=0.3, s=15, marker='o')
            
        outliers = raw_data_w_outliers[raw_data_w_outliers['is_outlier']]
        if not outliers.empty:
            ax.scatter(outliers['t'], outliers['y'], color='red', marker='x', s=50, label=TEXTS['legend_outlier'][lang], zorder=5)
            
        ax.errorbar(stats_df['t_round'], stats_df['mean'], yerr=stats_df['std'], 
                    fmt='o', color='black', ecolor='black', capsize=3, label=TEXTS['legend_mean'][lang], zorder=4)
        
        t_max_val = raw_data_w_outliers['t'].max()
        t_smooth = np.linspace(0, t_max_val, 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label=TEXTS['legend_global'][lang])
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            y_ind = model_func(t_smooth, y_i, y_f, ph['p'], ph['r_max'], ph['lambda'])
            y_vis = y_i + (y_f - y_i) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.6, label=TEXTS['legend_phase'][lang].format(i+1))
        
        ax.set_xlabel(TEXTS['axis_time'][lang])
        ax.set_ylabel(y_label)
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.3)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        st.download_button(
            label=TEXTS['download_plot'][lang],
            data=buf.getvalue(),
            file_name=f"plot_{n}_phases.svg",
            mime="image/svg+xml",
            key=f"dl_btn_{model_func.__name__}_{n}"
        )
        st.pyplot(fig)
        
    with c_data:
        df_glob = pd.DataFrame({
            "Param": [yi_name, yf_name], "Valor": [y_i, y_f], "SE": [y_i_se, y_f_se]
        })
        st.dataframe(df_glob.style.format({"Valor": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "F": i+1,
                "p": ph['p'], "SE p": ph['SE p'],
                rate_label: ph['r_max'], f"SE {rate_label}": ph['r_max_se'],
                "λ": ph['lambda'], "SE λ": ph['lambda_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", "SE p": "{:.4f}",
            rate_label: "{:.4f}", f"SE {rate_label}": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)
        
        m = res['metrics']
        df_met = pd.DataFrame({
            "Métrica": ["R²", "R² Adj", "AICc", "BIC"],
            "Valor": [m['R2'], m['R2_adj'], m['AICc'], m['BIC']]
        })
        st.dataframe(df_met.style.format({"Valor": "{:.4f}"}), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Analysis")
    
    # --- Language Selector ---
    lang_sel = st.sidebar.selectbox("Language / Idioma", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_sel]
    
    st.title(TEXTS['title'][lang])
    st.sidebar.header(TEXTS['sidebar_settings'][lang])
    
    var_type_options = ["Genérico y(t)", "Produto P(t)", "Substrato S(t)", "Biomassa X(t)"]
    var_type = st.sidebar.selectbox(TEXTS['response_type'][lang], var_type_options)
    
    y_label, param_labels, rate_label = VAR_LABELS[var_type][lang]
    
    file = st.sidebar.file_uploader(TEXTS['upload_label'][lang], type=["csv", "xlsx"])
    max_phases = st.sidebar.number_input(TEXTS['max_phases'][lang], 1, 10, 5)
    
    if not file: 
        st.info(TEXTS['info_upload'][lang])
        st.stop()
    
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        t_flat, y_flat, replicates = process_data(df)
        
        if len(replicates) == 0:
            st.error(TEXTS['error_cols'][lang])
            st.stop()
            
        st.write(TEXTS['data_loaded'][lang].format(len(replicates), len(t_flat)))
        
    except Exception as e: 
        st.error(TEXTS['error_read'][lang].format(e))
        st.stop()
    
    if st.button(TEXTS['run_button'][lang]):
        st.divider()
        tab_g, tab_b = st.tabs([TEXTS['tab_gompertz'][lang], TEXTS['tab_boltzmann'][lang]])
        
        def run_model_loop(model_name, model_func, color):
            results_list = []
            
            for n in range(1, max_phases + 1):
                with st.expander(TEXTS['expanding'][lang].format(model_name, n), expanded=False):
                    with st.spinner(TEXTS['optimizing'][lang].format(n)):
                        res = fit_model_auto(t_flat, y_flat, model_func, n)
                        if res is None:
                            st.warning(TEXTS['warning_insufficient'][lang])
                            continue
                        
                        display_single_fit(res, replicates, model_func, color, y_label, param_labels, rate_label, lang)
                        results_list.append(res)
            
            if not results_list: return

            st.markdown(f"### {TEXTS['table_title'][lang]}")
            summary_data = []
            best_aicc = np.inf
            best_model_idx = -1
            
            for i, res in enumerate(results_list):
                m = res['metrics']
                summary_data.append({
                    "F": res['n_phases'],
                    "R²": m['R2'],
                    "R² Adj": m['R2_adj'],
                    "SSE": m['SSE'],
                    "AIC": m['AIC'],
                    "AICc": m['AICc'],
                    "BIC": m['BIC']
                })
                if m['AICc'] < best_aicc:
                    best_aicc = m['AICc']
                    best_model_idx = i
            
            df_summary = pd.DataFrame(summary_data)
            
            def highlight_best(row):
                if row['AICc'] == best_aicc:
                    return ['background-color: #d4edda; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            st.dataframe(df_summary.style.apply(highlight_best, axis=1).format({
                "R²": "{:.4f}", "R² Adj": "{:.4f}", "SSE": "{:.4f}", 
                "AIC": "{:.4f}", "AICc": "{:.4f}", "BIC": "{:.4f}"
            }), hide_index=True)
            
            best_n = results_list[best_model_idx]['n_phases']
            st.success(TEXTS['best_model_msg'][lang].format(best_n))
            
            st.markdown(f"### {TEXTS['graph_summary_title'][lang]}")
            plot_metrics_summary(results_list, model_name, lang)

        with tab_g:
            run_model_loop("Gompertz", gompertz_term_eq32, "tab:blue")
            
        with tab_b:
            run_model_loop("Boltzmann", boltzmann_term_eq31, "tab:orange")

if __name__ == "__main__":
    main()
