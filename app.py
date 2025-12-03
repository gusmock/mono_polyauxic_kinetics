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
import io
import re
import os
from datetime import datetime

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

# English is the first key, making it the default in streamlit selectbox
LANGUAGES = {
    "English": "en",
    "Português (BR)": "pt",
    "Français (CA)": "fr"
}

TEXTS = {
    "app_title": {
        "en": "Polyauxic Modeling Platform",
        "pt": "Plataforma de Modelagem Poliauxica",
        "fr": "Plateforme de Modélisation Polyauxique"
    },
    "intro_desc": {
        "en": "This application performs advanced non-linear regression for microbial growth kinetics. It is designed to identify and fit mono- and polyauxic behaviors using robust statistical methods (Lorentzian loss, ROUT outlier detection) and provides model selection based on Information Criteria (AIC, BIC).",
        "pt": "Este aplicativo realiza regressão não-linear avançada para cinética de crescimento microbiano. Ele foi projetado para identificar e ajustar comportamentos mono e poliauxicos usando métodos estatísticos robustos (perda Lorentziana, detecção de outliers ROUT) e fornece seleção de modelos baseada em Critérios de Informação (AIC, BIC).",
        "fr": "Cette application effectue une régression non linéaire avancée pour la cinétique de croissance microbienne. Elle est conçue pour identifier et ajuster les comportements mono- et polyauxiques à l'aide de méthodes statistiques robustes (perte Lorentzienne, détection de valeurs aberrantes ROUT) et fournit une sélection de modèles basée sur les critères d'information (AIC, BIC)."
    },
    "paper_ref": {
        "en": "Reference Paper:",
        "pt": "Artigo de Referência:",
        "fr": "Article de Référence :"
    },
    "db_notice": {
        "en": "⚠️ Note: This program logs usage data to build a database for future reference and improvement. Your personal data is stored securely.",
        "pt": "⚠️ Nota: Este programa registra dados de uso para formar um banco de dados para referência futura e melhorias. Seus dados pessoais são armazenados com segurança.",
        "fr": "⚠️ Remarque : Ce programme enregistre les données d'utilisation pour constituer une base de données pour référence future. Vos données personnelles sont stockées en toute sécurité."
    },
    "instructions_header": {
        "en": "General Instructions",
        "pt": "Instruções Gerais",
        "fr": "Instructions Générales"
    },
    "instructions_list": {
        "en": """
        * **File Format:** CSV or Excel (.xlsx).
        * **Structure:** The first row must be the column header.
        * **Columns:** Arrange data in pairs (Time, Response). Example: Col A (Time 1), Col B (Response 1)...
        * **Replicates:** Accepts up to quintuplicates (5 replicates). The system automatically detects pairs.
        """,
        "pt": """
        * **Formato do Arquivo:** CSV ou Excel (.xlsx).
        * **Estrutura:** A primeira linha deve ser o rótulo da coluna.
        * **Colunas:** Organize os dados em pares (Tempo, Resposta). Exemplo: Col A (Tempo 1), Col B (Resposta 1)...
        * **Réplicas:** Aceita até quintuplicatas (5 réplicas). O sistema detecta os pares automaticamente.
        """,
        "fr": """
        * **Format de Fichier:** CSV ou Excel (.xlsx).
        * **Structure:** La première ligne doit être l'en-tête de la colonne.
        * **Colonnes:** Disposez les données par paires (Temps, Réponse). Exemple : Col A (Temps 1), Col B (Réponse 1)...
        * **Réplicats:** Accepte jusqu'à cinq réplicats. Le système détecte automatiquement les paires.
        """
    },
    "form_header": {"en": "User Identification (Mandatory)", "pt": "Identificação do Usuário (Obrigatório)", "fr": "Identification de l'Utilisateur (Obligatoire)"},
    "lbl_name": {"en": "Full Name", "pt": "Nome Completo", "fr": "Nom Complet"},
    "lbl_email": {"en": "E-mail", "pt": "E-mail", "fr": "Courriel"},
    "lbl_inst": {"en": "Institution", "pt": "Instituição de Origem", "fr": "Institution d'Origine"},
    "lbl_desc": {"en": "Description of Data to be Fitted", "pt": "Descrição dos Dados a serem Ajustados", "fr": "Description des Données à Ajuster"},
    "btn_start": {"en": "START ANALYSIS", "pt": "INICIAR ANÁLISE", "fr": "LANCER L'ANALYSE"},
    "err_fields": {"en": "Please fill in all fields.", "pt": "Por favor, preencha todos os campos.", "fr": "Veuillez remplir tous les champs."},
    "err_email": {"en": "Invalid e-mail format.", "pt": "Formato de e-mail inválido.", "fr": "Format d'e-mail invalide."},
    
    # --- ANALYSIS PAGE ---
    "sidebar_config": {"en": "Settings", "pt": "Configurações", "fr": "Paramètres"},
    "var_type": {"en": "Response Type (Y Axis)", "pt": "Tipo de Resposta (Eixo Y)", "fr": "Type de Réponse (Axe Y)"},
    "upload": {"en": "Upload Data File", "pt": "Carregar Arquivo de Dados", "fr": "Télécharger le Fichier de Données"},
    "max_phases": {"en": "Max Phases", "pt": "Máximo de Fases", "fr": "Phases Max"},
    "run_fit": {"en": "RUN MODEL FITTING", "pt": "EXECUTAR AJUSTE DO MODELO", "fr": "LANCER L'AJUSTEMENT DU MODÈLE"},
    "tab_gompertz": {"en": "Gompertz (Eq. 32)", "pt": "Gompertz (Eq. 32)", "fr": "Gompertz (Eq. 32)"},
    "tab_boltzmann": {"en": "Boltzmann (Eq. 31)", "pt": "Boltzmann (Eq. 31)", "fr": "Boltzmann (Eq. 31)"},
    "legend_outlier": {"en": "Outliers", "pt": "Outliers", "fr": "Valeurs Aberrantes"},
    "legend_mean": {"en": "Mean (w/o Outliers)", "pt": "Média (s/ Outliers)", "fr": "Moyenne (sans Aberrants)"},
    "legend_global": {"en": "Global Fit", "pt": "Ajuste Global", "fr": "Ajustement Global"},
    "legend_phase": {"en": "Phase {0}", "pt": "Fase {0}", "fr": "Phase {0}"},
    "axis_time": {"en": "Time (h/d)", "pt": "Tempo (h/d)", "fr": "Temps (h/j)"},
    "download_plot": {"en": "Download Plot (SVG)", "pt": "Baixar Gráfico (SVG)", "fr": "Télécharger le Graphique (SVG)"},
    "download_summary": {"en": "Download Summary (SVG)", "pt": "Baixar Resumo (SVG)", "fr": "Télécharger le Résumé (SVG)"},
    "best_model": {
        "en": "🏆 Best Suggested Model: **{0} Phase(s)** (Based on lowest AICc).",
        "pt": "🏆 Melhor Modelo Sugerido: **{0} Fase(s)** (Baseado no menor AICc).",
        "fr": "🏆 Meilleur Modèle Suggéré: **{0} Phase(s)** (Basé sur le plus bas AICc)."
    },
    "summary_title": {"en": "Effect of Phase Count on Criteria", "pt": "Efeito do Número de Fases nos Critérios", "fr": "Effet du Nombre de Phases sur les Critères"},
    "table_title": {"en": "Model Selection Table", "pt": "Tabela de Seleção de Modelo", "fr": "Tableau de Sélection du Modèle"},
    "data_loaded": {
        "en": "**Data Loaded:** {0} replicates identified. Total points: {1}",
        "pt": "**Dados Carregados:** {0} réplicas identificadas. Total de pontos: {1}",
        "fr": "**Données Chargées:** {0} réplicats identifiés. Points totaux: {1}"
    },
    "error_proc": {"en": "Error processing data: {0}", "pt": "Erro ao processar dados: {0}", "fr": "Erreur de traitement: {0}"},
    "warning_insuf": {"en": "Insufficient data.", "pt": "Dados insuficientes.", "fr": "Données insuffisantes."}
}

# --- Variable Labels Configuration ---
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
# 2. LOGIC FOR DATA STORAGE & VALIDATION
# ==============================================================================

def validate_email(email):
    """Regex validation for email."""
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

def log_user_access(name, email, inst, desc):
    """Saves user info to a local CSV."""
    file_path = "user_log.csv"
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Name": [name],
        "Email": [email],
        "Institution": [inst],
        "DataDescription": [desc]
    }
    df_new = pd.DataFrame(data)
    
    if not os.path.isfile(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode='a', header=False, index=False)

# ==============================================================================
# 3. MATHEMATICAL CORE (UNCHANGED LOGIC)
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Boltzmann model term (Eq. 31)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Gompertz model term (Eq. 32)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9: delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = r_max_j * np.e * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 1.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe * np.exp(-np.exp(exponent))

def polyauxic_model(t, theta, model_func, n_phases):
    """Global polyauxic model summation."""
    t = np.asarray(t, dtype=float)
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2+n_phases]
    r_max = theta[2+n_phases : 2+2*n_phases]
    lambda_ = theta[2+2*n_phases : 2+3*n_phases]
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases

def sse_loss(theta, t, y, model_func, n_phases):
    """Sum of Squared Errors Loss function."""
    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Numerical Hessian calculation."""
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
    """ROUT-based outlier detection."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def smart_initial_guess(t, y, n_phases):
    """Initial parameter guessing based on derivatives."""
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

def calculate_p_errors(z_vals, cov_z):
    """Standard error calculation for p (Softmax)."""
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
    """Main fitting function."""
    n_params = 2 + 3 * n_phases
    if len(t_data) <= n_params: return None 
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

    res_de = differential_evolution(sse_loss, bounds, args=(t_norm, y_norm, model_func, n_phases),
                                    maxiter=3000, popsize=pop_size, init=init_pop, strategy='best1bin',
                                    seed=None, polish=True, tol=1e-6)
    res_opt = minimize(sse_loss, res_de.x, args=(t_norm, y_norm, model_func, n_phases),
                       method='L-BFGS-B', bounds=bounds, tol=1e-10)
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
        n_obs = len(y_norm); n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2:2+n_phases] = se_norm[2:2+n_phases] 
        se_real[2+n_phases:2+2*n_phases] = se_norm[2+n_phases:2+2*n_phases] * scale_r
        se_real[2+2*n_phases:2+3*n_phases] = se_norm[2+2*n_phases:2+3*n_phases] * scale_l
        idx_z_start = 2; idx_z_end = 2 + n_phases
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
    n_len = len(y_data); k = len(theta_real)
    if sse <= 1e-12: sse = 1e-12
    if (n_len - k - 1) > 0: r2_adj = 1 - (1 - r2) * (n_len - 1) / (n_len - k - 1)
    else: r2_adj = np.nan
    aic = n_len * np.log(sse/n_len) + 2*k
    bic = n_len * np.log(sse/n_len) + k * np.log(n_len)
    aicc = aic + (2*k*(k+1))/(n_len-k-1) if (n_len-k-1)>0 else np.inf
    
    return {"n_phases": n_phases, "theta": theta_real, "se": se_real, "se_p": se_p,
            "metrics": {"R2": r2, "R2_adj": r2_adj, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
            "outliers": outliers, "y_pred": y_pred}

# ==============================================================================
# 4. DATA PROCESSING
# ==============================================================================

def process_data(df):
    """Processes replicate columns."""
    df = df.dropna(axis=1, how='all')
    cols = df.columns.tolist()
    all_t = []; all_y = []; replicates = []
    num_replicates = len(cols) // 2
    for i in range(num_replicates):
        t_col = cols[2*i]; y_col = cols[2*i+1]
        t_vals = pd.to_numeric(df[t_col], errors='coerce').values
        y_vals = pd.to_numeric(df[y_col], errors='coerce').values
        mask = ~np.isnan(t_vals) & ~np.isnan(y_vals)
        t_clean = t_vals[mask]; y_clean = y_vals[mask]
        all_t.extend(t_clean); all_y.extend(y_clean)
        replicates.append({'t': t_clean, 'y': y_clean, 'name': f'Replica {i+1}'})
    t_flat = np.array(all_t); y_flat = np.array(all_y)
    idx_sort = np.argsort(t_flat)
    return t_flat[idx_sort], y_flat[idx_sort], replicates

def calculate_mean_with_outliers(replicates, model_func, theta, n_phases):
    """Calculate statistics excluding outliers."""
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
# 5. VIEW COMPONENTS
# ==============================================================================

def plot_metrics_summary(results_list, model_name, lang):
    """Summary chart."""
    phases = [r['n_phases'] for r in results_list]
    aic = [r['metrics']['AIC'] for r in results_list]
    aicc = [r['metrics']['AICc'] for r in results_list]
    bic = [r['metrics']['BIC'] for r in results_list]
    r2_adj = [r['metrics']['R2_adj'] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(phases, aic, 'o--', label='AIC')
    ax1.plot(phases, aicc, 's-', label='AICc')
    ax1.plot(phases, bic, '^:', label='BIC')
    ax1.set_xlabel('Number of Phases')
    ax1.set_ylabel('Value')
    ax1.set_title('Information Criteria')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(phases, r2_adj, 'o-', color='purple', label='Adjusted R²')
    ax2.set_xlabel('Number of Phases')
    ax2.set_ylabel('Adjusted R²')
    ax2.set_title('Fit Quality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    st.download_button(label=TEXTS['download_summary'][lang], data=buf.getvalue(),
                       file_name=f"metrics_summary_{model_name}.svg", mime="image/svg+xml",
                       key=f"dl_summary_{model_name}")
    st.pyplot(fig)

def display_single_fit(res, replicates, model_func, color_main, y_label, param_labels, rate_label, lang):
    n = res['n_phases']; theta = res['theta']; se = res['se']; se_p = res['se_p']
    yi_name, yf_name = param_labels
    stats_df, raw_data_w_outliers = calculate_mean_with_outliers(replicates, model_func, theta, n)
    y_i, y_f = theta[0], theta[1]; y_i_se, y_f_se = se[0], se[1]
    
    z = theta[2:2+n]
    r_max = theta[2+n:2+2*n]; r_max_se = se[2+n:2+2*n]
    lambda_ = theta[2+2*n:2+3*n]; lambda_se = se[2+2*n:2+3*n]
    p = np.exp(z - np.max(z)); p /= np.sum(p)
    
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
        
        ax.set_xlabel(TEXTS['axis_time'][lang]); ax.set_ylabel(y_label)
        ax.legend(fontsize='small'); ax.grid(True, linestyle=':', alpha=0.3)
        
        buf = io.BytesIO(); fig.savefig(buf, format="svg")
        st.download_button(label=TEXTS['download_plot'][lang], data=buf.getvalue(),
                           file_name=f"plot_{n}_phases.svg", mime="image/svg+xml", key=f"dl_btn_{model_func.__name__}_{n}")
        st.pyplot(fig)
        
    with c_data:
        df_glob = pd.DataFrame({"Param": [yi_name, yf_name], "Val": [y_i, y_f], "SE": [y_i_se, y_f_se]})
        st.dataframe(df_glob.style.format({"Val": "{:.4f}", "SE": "{:.4f}"}), hide_index=True)
        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                "F": i+1, "p": ph['p'], "SE p": ph['SE p'],
                rate_label: ph['r_max'], f"SE {rate_label}": ph['r_max_se'],
                "λ": ph['lambda'], "SE λ": ph['lambda_se']
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "p": "{:.4f}", "SE p": "{:.4f}", rate_label: "{:.4f}", f"SE {rate_label}": "{:.4f}",
            "λ": "{:.4f}", "SE λ": "{:.4f}"
        }), hide_index=True)
        m = res['metrics']
        df_met = pd.DataFrame({"Metric": ["R²", "R² Adj", "AICc", "BIC"], "Value": [m['R2'], m['R2_adj'], m['AICc'], m['BIC']]})
        st.dataframe(df_met.style.format({"Value": "{:.4f}"}), hide_index=True)

# ==============================================================================
# 6. APP STRUCTURE
# ==============================================================================

def show_home_page(lang):
    st.markdown(f"### {TEXTS['app_title'][lang]}")
    st.info(TEXTS['intro_desc'][lang])
    st.markdown(f"**{TEXTS['paper_ref'][lang]}** [https://doi.org/10.48550/arXiv.2507.05960](https://doi.org/10.48550/arXiv.2507.05960)")
    st.warning(TEXTS['db_notice'][lang])
    with st.expander(TEXTS['instructions_header'][lang], expanded=False):
        st.markdown(TEXTS['instructions_list'][lang])
    st.markdown("---")
    st.subheader(TEXTS['form_header'][lang])
    with st.form("user_login_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input(TEXTS['lbl_name'][lang])
        email = c2.text_input(TEXTS['lbl_email'][lang])
        inst = st.text_input(TEXTS['lbl_inst'][lang])
        desc = st.text_area(TEXTS['lbl_desc'][lang])
        submitted = st.form_submit_button(TEXTS['btn_start'][lang])
        if submitted:
            if not name or not email or not inst or not desc:
                st.error(TEXTS['err_fields'][lang])
            elif not validate_email(email):
                st.error(TEXTS['err_email'][lang])
            else:
                log_user_access(name, email, inst, desc)
                st.session_state.page = 'analysis'
                st.rerun()

def show_analysis_page(lang):
    st.markdown(f"## {TEXTS['app_title'][lang]}")
    st.sidebar.header(TEXTS['sidebar_config'][lang])
    var_type_opts = list(VAR_LABELS.keys())
    var_type = st.sidebar.selectbox(TEXTS['var_type'][lang], var_type_opts)
    y_label, param_labels, rate_label = VAR_LABELS[var_type][lang]
    file = st.sidebar.file_uploader(TEXTS['upload'][lang], type=["csv", "xlsx"])
    max_phases = st.sidebar.number_input(TEXTS['max_phases'][lang], 1, 10, 5)
    
    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            t_flat, y_flat, replicates = process_data(df)
            if not replicates:
                st.error(TEXTS['error_cols'][lang])
            else:
                st.success(TEXTS['data_loaded'][lang].format(len(replicates), len(t_flat)))
                if st.button(TEXTS['run_fit'][lang]):
                    st.divider()
                    tab1, tab2 = st.tabs([TEXTS['tab_gompertz'][lang], TEXTS['tab_boltzmann'][lang]])
                    for tab, model_name, func, color in [
                        (tab1, "Gompertz", gompertz_term_eq32, "tab:blue"),
                        (tab2, "Boltzmann", boltzmann_term_eq31, "tab:orange")
                    ]:
                        with tab:
                            results_list = []
                            for n in range(1, max_phases + 1):
                                with st.expander(TEXTS['expanding'][lang].format(model_name, n), expanded=False):
                                    with st.spinner(TEXTS['optimizing'][lang].format(n)):
                                        res = fit_model_auto(t_flat, y_flat, func, n)
                                        if res:
                                            display_single_fit(res, replicates, func, color, y_label, param_labels, rate_label, lang)
                                            results_list.append(res)
                                        else:
                                            st.warning(TEXTS['warning_insufficient'][lang])
                            if results_list:
                                st.markdown(f"### {TEXTS['table_title'][lang]}")
                                summary_data = []
                                best_aicc = np.inf; best_idx = 0
                                for i, r in enumerate(results_list):
                                    m = r['metrics']
                                    summary_data.append({
                                        "F": r['n_phases'], "R²": m['R2'], "R² Adj": m['R2_adj'],
                                        "SSE": m['SSE'], "AIC": m['AIC'], "AICc": m['AICc'], "BIC": m['BIC']
                                    })
                                    if m['AICc'] < best_aicc:
                                        best_aicc = m['AICc']; best_idx = i
                                st.dataframe(pd.DataFrame(summary_data).style.apply(
                                    lambda x: ['background-color: #d4edda; font-weight: bold' if x['AICc'] == best_aicc else '' for _ in x], 
                                    axis=1).format("{:.4f}"), hide_index=True)
                                st.success(TEXTS['best_model'][lang].format(results_list[best_idx]['n_phases']))
                                st.markdown(f"### {TEXTS['summary_title'][lang]}")
                                plot_metrics_summary(results_list, model_name, lang)
        except Exception as e:
            st.error(TEXTS['error_proc'][lang].format(e))
    else:
        st.info(TEXTS['info_upload'][lang])

def main():
    if 'page' not in st.session_state: st.session_state.page = 'home'
    lang_key = st.sidebar.selectbox("Language / Idioma / Langue", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_key]
    if st.session_state.page == 'home': show_home_page(lang)
    else: show_analysis_page(lang)

if __name__ == "__main__":
    main()
