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
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
import re
import os
from datetime import datetime

# ==============================================================================
# 0. MULTILINGUAL DICTIONARY & CONFIG
# ==============================================================================

LANGUAGES = {
    "English": "en",
    "Português (BR)": "pt",
    "Français (CA)": "fr"
}

TEXTS = {
    "title": {
        "en": "Polyauxic Modeling (Replicates & Metadata)",
        "pt": "Modelagem Poliauxica (Réplicas e Metadados)",
        "fr": "Modélisation Polyauxique (Réplicats et Métadonnées)"
    },
    "sidebar_user": {
        "en": "User Information",
        "pt": "Informações do Usuário",
        "fr": "Informations de l'Utilisateur"
    },
    "name": {"en": "Full Name", "pt": "Nome Completo", "fr": "Nom Complet"},
    "institution": {"en": "Institution", "pt": "Instituição", "fr": "Institution"},
    "email": {"en": "E-mail", "pt": "E-mail", "fr": "Courriel"},
    "desc": {"en": "Data Description", "pt": "Descrição dos Dados", "fr": "Description des Données"},
    "invalid_email": {
        "en": "Please enter a valid email.",
        "pt": "Por favor, insira um e-mail válido.",
        "fr": "Veuillez entrer un courriel valide."
    },
    "sidebar_config": {"en": "Settings", "pt": "Configurações", "fr": "Paramètres"},
    "var_type": {"en": "Response Type (Y Axis)", "pt": "Tipo de Resposta (Eixo Y)", "fr": "Type de Réponse (Axe Y)"},
    "upload": {
        "en": "Upload CSV/XLSX (Pairs: t1, y1, t2, y2...)",
        "pt": "Arquivo CSV/XLSX (Pares: t1, y1, t2, y2...)",
        "fr": "Télécharger CSV/XLSX (Paires: t1, y1, t2, y2...)"
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
    "run_btn": {"en": "RUN COMPARATIVE ANALYSIS", "pt": "EXECUTAR ANÁLISE COMPARATIVA", "fr": "LANCER L'ANALYSE COMPARATIVE"},
    "error_cols": {
        "en": "Could not identify column pairs. Check file format.",
        "pt": "Não foi possível identificar pares de colunas. Verifique o arquivo.",
        "fr": "Impossible d'identifier les paires de colonnes. Vérifiez le fichier."
    },
    "error_proc": {"en": "Processing error: {0}", "pt": "Erro ao processar dados: {0}", "fr": "Erreur de traitement: {0}"},
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
        "en": "Insufficient data for this number of phases.",
        "pt": "Dados insuficientes para este número de fases.",
        "fr": "Données insuffisantes pour ce nombre de phases."
    },
    "table_title": {
        "en": "Model Selection Table (Information Criteria)",
        "pt": "Tabela de Seleção de Modelo (Critérios de Informação)",
        "fr": "Tableau de Sélection du Modèle (Critères d'Information)"
    },
    "best_model": {
        "en": "🏆 Best Suggested Model: **{0} Phase(s)** (Based on lowest AICc).",
        "pt": "🏆 Melhor Modelo Sugerido: **{0} Fase(s)** (Baseado no menor AICc).",
        "fr": "🏆 Meilleur Modèle Suggéré: **{0} Phase(s)** (Basé sur le plus bas AICc)."
    },
    "graph_legend_valid": {"en": "Valid Data", "pt": "Dados Válidos", "fr": "Données Valides"},
    "graph_legend_outlier": {"en": "Outliers", "pt": "Outliers", "fr": "Valeurs Aberrantes"},
    "graph_legend_mean": {"en": "Mean (w/o Outliers)", "pt": "Média (s/ Outliers)", "fr": "Moyenne (sans Aberrants)"},
    "graph_legend_global": {"en": "Global Fit", "pt": "Ajuste Global", "fr": "Ajustement Global"},
    "graph_legend_phase": {"en": "Phase {0}", "pt": "Fase {0}", "fr": "Phase {0}"},
    "axis_time": {"en": "Time (h/d)", "pt": "Tempo (h/d)", "fr": "Temps (h/j)"},
    "db_saved": {"en": "Data saved to database.", "pt": "Dados salvos no banco de dados.", "fr": "Données enregistrées dans la base."}
}

LABELS_MAP = {
    "Genérico y(t)": {
        "en": "Response (y)", "pt": "Resposta (y)", "fr": "Réponse (y)",
        "params": ("y_i", "y_f"), "rate": "r_max"
    },
    "Produto P(t)": {
        "en": "Product Concentration (P)", "pt": "Concentração de Produto (P)", "fr": "Concentration en Produit (P)",
        "params": ("P_i", "P_f"), "rate": "r_P,max"
    },
    "Substrato S(t)": {
        "en": "Substrate Concentration (S)", "pt": "Concentração de Substrato (S)", "fr": "Concentration en Substrat (S)",
        "params": ("S_i", "S_f"), "rate": "r_S,max"
    },
    "Biomassa X(t)": {
        "en": "Cell Concentration (X)", "pt": "Concentração Celular (X)", "fr": "Concentration Cellulaire (X)",
        "params": ("X_i", "X_f"), "rate": "µ_max"
    }
}

def get_text(key, lang_code):
    """Helper to retrieve text based on language code."""
    return TEXTS[key][lang_code].format() if key in TEXTS else key

# ==============================================================================
# 1. MATHEMATICAL MODELS (EXACT NOTATION EQS. 31 & 32)
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Phase j term for the Boltzmann model (Eq. 31).
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """
    Phase j term for the Gompertz model (Eq. 32).
    """
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)

    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)

    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """
    Global Model: Weighted sum of sigmoidal phases.
    """
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    
    z = theta[2 : 2+n_phases]
    r_max = theta[2+n_phases : 2+2*n_phases]
    lambda_ = theta[2+2*n_phases : 2+3*n_phases]

    # Softmax for weights
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])

    return y_i + (y_f - y_i) * sum_phases

# ==============================================================================
# 2. LOSS FUNCTIONS AND STATISTICS
# ==============================================================================

def sse_loss(theta, t, y, model_func, n_phases):
    """
    Objective Function: Sum of Squared Errors (SSE).
    """
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    # Slight physical penalty for negative values if not expected
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): 
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
    """Visual method to mark outliers (ROUT-like)."""
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

def fit_model_auto(t_data, y_data, model_func, n_phases):
    
    # Degrees of freedom validation
    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params:
        return None 

    # 1. Normalization
    t_scale = np.max(t_data) if np.max(t_data) > 0 else 1.0
    y_scale = np.max(y_data) if np.max(y_data) > 0 else 1.0
    t_norm = t_data / t_scale
    y_norm = y_data / y_scale
    
    # 2. Initialization
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

    # 3. Global Optimization (SSE/R2 focused)
    res_de = differential_evolution(
        sse_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
        seed=None, polish=True, tol=1e-6
    )
    
    # 4. Local Refinement
    res_opt = minimize(
        sse_loss, res_de.x, args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B', bounds=bounds, tol=1e-10
    )
    
    theta_norm = res_opt.x
    
    # 5. Denormalization and Errors
    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)

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
    except:
        se_real = np.full_like(theta_real, np.nan)

    # 6. Calculate Information Criteria
    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)
    
    sse = np.sum((y_data - y_pred)**2)
    sst = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - sse/sst
    
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12
    
    # Adjusted R2
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
        "metrics": {"R2": r2, "R2_adj": r2_adj, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

# ==============================================================================
# 4. DATA PROCESSING (REPLICATES & STORAGE)
# ==============================================================================

def validate_email(email):
    """Simple regex email validation."""
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

def save_user_data(name, inst, email, desc, filename):
    """Saves user metadata to a local CSV file."""
    db_file = "polyauxic_user_database.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_data = pd.DataFrame([{
        "Timestamp": timestamp,
        "Name": name,
        "Institution": inst,
        "Email": email,
        "Description": desc,
        "UploadedFile": filename
    }])
    
    if os.path.exists(db_file):
        new_data.to_csv(db_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(db_file, mode='w', header=True, index=False)

def process_data(df):
    """
    Processes DataFrame detecting replicates in pairs of columns.
    """
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
# 5. INTERFACE AND VISUALIZATION
# ==============================================================================

def display_single_fit(res, replicates, model_func, color_main, y_label, param_labels, rate_label, lang):
    n = res['n_phases']
    theta = res['theta']
    se = res['se']
    
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
            "p": p[i],
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
            ax.scatter(outliers['t'], outliers['y'], color='red', marker='x', s=50, label=TEXTS['graph_legend_outlier'][lang], zorder=5)
            
        ax.errorbar(stats_df['t_round'], stats_df['mean'], yerr=stats_df['std'], 
                    fmt='o', color='black', ecolor='black', capsize=3, label=TEXTS['graph_legend_mean'][lang], zorder=4)
        
        t_max_val = raw_data_w_outliers['t'].max()
        t_smooth = np.linspace(0, t_max_val, 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        
        # No error shading as requested
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label=TEXTS['graph_legend_global'][lang])
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            y_ind = model_func(t_smooth, y_i, y_f, ph['p'], ph['r_max'], ph['lambda'])
            y_vis = y_i + (y_f - y_i) * y_ind
            ax.plot(t_smooth, y_vis, '--', color=colors[i], alpha=0.6, label=TEXTS['graph_legend_phase'][lang].format(i+1))
        
        ax.set_xlabel(TEXTS['axis_time'][lang])
        ax.set_ylabel(y_label)
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.3)
        st.pyplot(fig)
        
    with c_data:
        df_glob = pd.DataFrame({
            "Param": [yi_name, yf_name], "Val": [y_i, y_f], "SE": [y_i_se, y_f_se]
        })
        st.dataframe(df_glob.style.format({"Val": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "F": i+1,
                "p": ph['p'],
                rate_label: ph['r_max'], f"SE {rate_label}": ph['r_max_se'],
                "λ": ph['lambda'], "SE λ": ph['lambda_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", rate_label: "{:.4f}", f"SE {rate_label}": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)
        
        m = res['metrics']
        df_met = pd.DataFrame({
            "Metric": ["R²", "R² Adj", "AICc", "BIC"],
            "Value": [m['R2'], m['R2_adj'], m['AICc'], m['BIC']]
        })
        st.dataframe(df_met.style.format({"Value": "{:.4f}"}), hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Modeling")
    
    # --- LANGUAGE SELECTION ---
    lang_name = st.sidebar.selectbox("Language / Idioma / Langue", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_name]
    
    st.title(TEXTS['title'][lang])
    
    # --- USER INFO FORM ---
    st.sidebar.header(TEXTS['sidebar_user'][lang])
    
    with st.sidebar.form("user_form"):
        u_name = st.text_input(TEXTS['name'][lang])
        u_inst = st.text_input(TEXTS['institution'][lang])
        u_email = st.text_input(TEXTS['email'][lang])
        u_desc = st.text_area(TEXTS['desc'][lang])
        submitted = st.form_submit_button("Save User Info")
    
    if submitted:
        if validate_email(u_email):
            st.sidebar.success("OK")
        else:
            st.sidebar.error(TEXTS['invalid_email'][lang])

    # --- CONFIGURATION ---
    st.sidebar.header(TEXTS['sidebar_config'][lang])
    
    var_type_options = ["Genérico y(t)", "Produto P(t)", "Substrato S(t)", "Biomassa X(t)"]
    var_type = st.sidebar.selectbox(TEXTS['var_type'][lang], var_type_options)
    
    # Translations for Y-axis labels and parameters
    labels_db = {
        "Genérico y(t)": {
            "en": ("Response (y)", ("y_i", "y_f"), "r_max"),
            "pt": ("Resposta (y)", ("y_i", "y_f"), "r_max"),
            "fr": ("Réponse (y)", ("y_i", "y_f"), "r_max")
        },
        "Produto P(t)": {
            "en": ("Product Concentration (P)", ("P_i", "P_f"), "r_P,max"),
            "pt": ("Concentração de Produto (P)", ("P_i", "P_f"), "r_P,max"),
            "fr": ("Concentration en Produit (P)", ("P_i", "P_f"), "r_P,max")
        },
        "Substrato S(t)": {
            "en": ("Substrate Concentration (S)", ("S_i", "S_f"), "r_S,max"),
            "pt": ("Concentração de Substrato (S)", ("S_i", "S_f"), "r_S,max"),
            "fr": ("Concentration en Substrat (S)", ("S_i", "S_f"), "r_S,max")
        },
        "Biomassa X(t)": {
            "en": ("Cell Concentration (X)", ("X_i", "X_f"), "µ_max"),
            "pt": ("Concentração Celular (X)", ("X_i", "X_f"), "µ_max"),
            "fr": ("Concentration Cellulaire (X)", ("X_i", "X_f"), "µ_max")
        }
    }
    
    y_label, param_labels, rate_label = labels_db[var_type][lang]
    
    file = st.sidebar.file_uploader(TEXTS['upload'][lang], type=["csv", "xlsx"])
    max_phases = st.sidebar.number_input(TEXTS['max_phases'][lang], 1, 6, 5)
    
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
        st.error(TEXTS['error_proc'][lang].format(e))
        st.stop()
    
    if st.button(TEXTS['run_btn'][lang]):
        
        # Save data if email is valid
        if validate_email(u_email):
            save_user_data(u_name, u_inst, u_email, u_desc, file.name)
            st.toast(TEXTS['db_saved'][lang])
        elif u_email:
            st.error(TEXTS['invalid_email'][lang])
            st.stop()

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
            st.success(TEXTS['best_model'][lang].format(best_n))

        with tab_g:
            run_model_loop("Gompertz", gompertz_term_eq32, "tab:blue")
            
        with tab_b:
            run_model_loop("Boltzmann", boltzmann_term_eq31, "tab:orange")

if __name__ == "__main__":
    main()
