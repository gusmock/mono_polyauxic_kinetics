"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 

-------------------------------------------------------------------------------
POLYAUXIC MODELING PLATFORM: COMPUTATIONAL WORKFLOW & THEORETICAL FRAMEWORK
-------------------------------------------------------------------------------

1. DATA INGESTION & NORMALIZATION
   - The user uploads experimental data (CSV/XLSX). Time (t) and response (y) are 
     normalized by their maximum values to ensure numerical stability during 
     optimization.
   - Reference: Mockaitis, G. (2025). Mono and Polyauxic Growth Kinetic Models. 
     ArXiv: 2507.05960.

2. MATHEMATICAL MODELS (Sigmoidal Functions)
   - The platform utilizes unified, mechanistic reparameterizations of classic 
     sigmoidal equations.
   - **Boltzmann:** Originally for phase transitions [Boltzmann, 1872], adapted 
     here to define max rate (r_max) and lag time (lambda) [Eq. 31].
   - **Gompertz:** Originally for mortality [Gompertz, 1825], widely used in 
     microbiology [Zwietering et al., 1990], reparameterized for asymmetry [Eq. 32].
   - **Polyauxic Growth:** Modeled as a weighted sum of 'n' monoauxic phases 
     [Mockaitis et al., 2020]. The weighting factors (p_j) are constrained via 
     a Softmax transformation [Eq. 33] to ensure they sum to 1.

3. HEURISTIC INITIALIZATION
   - To avoid local minima, initial estimates are derived automatically by analyzing 
     peaks in the first derivative (dy/dt) of the data. This provides biologically 
     plausible starting points for the rates (r_max) and inflection points (lambda).

4. ROBUST FITTING STRATEGY (Two-Stage Optimization)
   - **Stage 1 (Global):** Differential Evolution (DE) [Storn & Price, 1997]. A 
     stochastic, population-based method used to explore the global parameter space 
     and avoid local optima common in multi-modal polyauxic landscapes.
   - **Stage 2 (Local):** L-BFGS-B [Byrd et al., 1995]. A gradient-based algorithm 
     applied to refine the parameters with high precision, strictly enforcing bound 
     constraints (e.g., non-negative rates).

5. OUTLIER DETECTION (ROUT Method)
   - Based on the premise that biological data often follow a heavy-tailed distribution due 
     to experimental error.
   - **Pre-fit:** Uses a Robust Loss Function (Charbonnier/Soft L1) [Charbonnier et al., 1994] 
     instead of Least Squares to minimize the influence of extreme points.
   - **Detection:** Applies the ROUT method [Motulsky & Brown, 2006], utilizing the 
     False Discovery Rate (FDR) [Benjamini & Hochberg, 1995] to identify outliers 
     statistically inconsistent with the model.

6. MODEL SELECTION (Parsimony)
   - To prevent overparameterization, the system fits models from 1 to 'n' phases.
   - Selection is guided by Information Criteria:
     - **AIC (Akaike):** [Akaike, 1974]
     - **AICc (Corrected):** For small sample sizes [Hurvich & Tsai, 1989].
     - **BIC (Bayesian):** For larger datasets, prioritizing parsimony [Schwarz, 1978].

7. UNCERTAINTY QUANTIFICATION
   - Parameter uncertainties (Standard Errors) are estimated via the Hessian matrix 
     (calculated numerically) and the residual variance.
   - Errors for derived parameters (like phase weights) are propagated using the 
     Delta Method [Oehlert, 1992].

-------------------------------------------------------------------------------
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import t as t_dist
import io

# ==============================================================================
# 0. CONFIGURATION & GLOBAL SETTINGS
# ==============================================================================

# Global Plot Style - Times New Roman 11
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 12,
    'mathtext.fontset': 'stix'
})

# Languages Configuration
LANGUAGES = {
    "🇬🇧 English": "en",
    "🇧🇷 Português (BR)": "pt",
    "🇫🇷 Français (CA)": "fr"
}

# UI Text Dictionary
TEXTS = {
    "zenodo_cite": {
        "en": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "pt": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "fr": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Logiciel]. Zenodo."
    },
    "app_title": {
        "en": "Polyauxic Modeling Platform",
        "pt": "Plataforma de Modelagem Poliauxica",
        "fr": "Plateforme de Modélisation Polyauxique"
    },
    "intro_desc": {
        "en": "This application performs advanced non-linear regression for microbial growth kinetics. It identifies mono- and polyauxic behaviors using robust statistical methods (Lorentzian loss, ROUT outlier detection) and selects models via Information Criteria (AIC, AICc, BIC).",
        "pt": "Este aplicativo realiza regressão não-linear avançada para cinética microbiana. Identifica comportamentos mono e poliauxicos usando métodos estatísticos robustos (perda Lorentziana, outliers ROUT) e seleciona modelos via Critérios de Informação (AIC, AICc, BIC).",
        "fr": "Cette application effectue une régression non linéaire avancée pour la cinétique microbienne. Elle identifie les comportements mono- et polyauxiques à l'aide de méthodes robustes (perte Lorentzienne, ROUT) et sélectionne les modèles via Critères d'Information (AIC, AICc, BIC)."
    },
    "paper_ref": {
        "en": "Reference Paper & Source:",
        "pt": "Artigo de Referência e Fonte:",
        "fr": "Article de Référence et Source :"
    },
    "instructions_header": {
        "en": "Instructions & File Format",
        "pt": "Instruções e Formato do Arquivo",
        "fr": "Instructions et Format de Fichier"
    },
    "instructions_list": {
        "en": """
        **Data Preparation & Format:**
        * **File Type:** Upload a `.csv` or `.xlsx` (Excel) file.
        * **Column Structure (Crucial):** Organize your data in **pairs** of columns: Time followed by Response.
        * **Headers:** The first row must contain the column names.
        * **Replicates:** You can include up to **5 biological replicates**. The system automatically detects them based on the column pairs.
        * **Decimals:** Both dot (`.`) and comma (`,`) are accepted.

        **Example Layout:**
        | A (Time 1) | B (Resp 1) | C (Time 2) | D (Resp 2) |
        | :--- | :--- | :--- | :--- |
        | 0.0 | 0.105 | 0.0 | 0.102 |
        | 1.0 | 0.200 | 1.0 | 0.198 |
        """,
        "pt": """
        **Preparação e Formato dos Dados:**
        * **Tipo de Arquivo:** Carregue um arquivo `.csv` ou `.xlsx` (Excel).
        * **Estrutura das Colunas (Importante):** Organize seus dados estritamente em **pares**: Tempo seguido de Resposta.
        * **Cabeçalho:** A primeira linha deve conter o nome das variáveis.
        * **Réplicas:** O sistema aceita até **5 réplicas biológicas**. Basta adicionar os pares de colunas lado a lado; o sistema os agrupará automaticamente.
        * **Decimais:** Tanto ponto (`.`) quanto vírgula (`,`) são aceitos.

        **Exemplo de Layout:**
        | A (Tempo 1) | B (Resp 1) | C (Tempo 2) | D (Resp 2) |
        | :--- | :--- | :--- | :--- |
        | 0.0 | 0.105 | 0.0 | 0.102 |
        | 1.0 | 0.200 | 1.0 | 0.198 |
        """,
        "fr": """
        **Préparation et Format des Données :**
        * **Type de Fichier :** Téléchargez un fichier `.csv` ou `.xlsx` (Excel).
        * **Structure des Colonnes (Important) :** Organisez vos données en **paires** : Temps suivi de Réponse.
        * **En-têtes :** La première ligne doit contenir les noms des colonnes.
        * **Réplicats :** Vous pouvez inclure jusqu'à **5 réplicats biologiques**. Le système les détecte automatiquement.
        * **Décimales :** Les points (`.`) et les virgules (`,`) sont acceptés.

        **Exemple de mise en page:**
        | A (Temps 1) | B (Rep 1) | C (Temps 2) | D (Rep 2) |
        | :--- | :--- | :--- | :--- |
        | 0.0 | 0.105 | 0.0 | 0.102 |
        | 1.0 | 0.200 | 1.0 | 0.198 |
        """
    },
    "sidebar_config": {"en": "Settings", "pt": "Configurações", "fr": "Paramètres"},
    "var_type": {"en": "Response Type (Y Axis)", "pt": "Tipo de Resposta (Eixo Y)", "fr": "Type de Réponse (Axe Y)"},
    "upload_label": {
        "en": "Upload CSV/XLSX (Col pairs: t1, y1, t2, y2...)",
        "pt": "Arquivo CSV/XLSX (Pares colunas: t1, y1, t2, y2...)",
        "fr": "Télécharger CSV/XLSX (Paires col: t1, y1, t2, y2...)"
    },
    "max_phases": {"en": "Max Phases to Test", "pt": "Máximo de Fases para testar", "fr": "Phases Max à Tester"},
    "info_upload": {
        "en": "Please upload a file to start.",
        "pt": "Por favor, carregue um arquivo para começar.",
        "fr": "Veuillez télécharger un fichier pour commencer."
    },
    "data_loaded": {
        "en": "Data Loaded: {0} replicates identified. Total points: {1}",
        "pt": "Dados Carregados: {0} réplicas identificadas. Total de pontos: {1}",
        "fr": "Données Chargées: {0} réplicats identifiés. Points totaux: {1}"
    },
    "run_button": {"en": "RUN ANALYSIS", "pt": "EXECUTAR ANÁLISE", "fr": "LANCER L'ANALYSE"},
    "tab_gompertz": {"en": "Gompertz (Eq. 32)", "pt": "Gompertz (Eq. 32)", "fr": "Gompertz (Eq. 32)"},
    "tab_boltzmann": {"en": "Boltzmann (Eq. 31)", "pt": "Boltzmann (Eq. 31)", "fr": "Boltzmann (Eq. 31)"},
    "expanding": {"en": "{0}: Fitting {1} Phase(s)", "pt": "{0}: Ajuste com {1} Fase(s)", "fr": "{0}: Ajustement avec {1} Phase(s)"},
    "optimizing": {"en": "Optimizing {0} phases...", "pt": "Otimizando {0} fases...", "fr": "Optimisation de {0} phases..."},
    "warning_insufficient": {"en": "Insufficient data.", "pt": "Dados insuficientes.", "fr": "Données insuffisantes."},
    "table_title": {"en": "Model Selection Table", "pt": "Tabela de Seleção de Modelo", "fr": "Tableau de Sélection du Modèle"},
    "best_model_msg": {
        "en": "🏆 Best Suggested Model: **{0} Phase(s)** (First local minimum of {1}).",
        "pt": "🏆 Melhor Modelo Sugerido: **{0} Fase(s)** (Primeiro mínimo local de {1}).",
        "fr": "🏆 Meilleur Modèle Suggéré : **{0} Phase(s)** (Premier minimum local de {1})."
    },
    "graph_summary_title": {
        "en": "Effect of Phase Count on Criteria",
        "pt": "Efeito do Número de Fases nos Critérios",
        "fr": "Effet du Nombre de Phases sur les Critères"
    },
    "download_plot": {"en": "Download Plot (SVG)", "pt": "Baixar Gráfico (SVG)", "fr": "Télécharger le Graphique (SVG)"},
    "download_summary": {"en": "Download Summary (SVG)", "pt": "Baixar Resumo (SVG)", "fr": "Télécharger le Résumé (SVG)"},
    "axis_time": {"en": "Time (h/d)", "pt": "Tempo (h/d)", "fr": "Temps (h/j)"},
    "legend_global": {"en": "Global Fit", "pt": "Ajuste Global", "fr": "Ajustement Global"},
    "legend_phase": {"en": "Phase {0}", "pt": "Fase {0}", "fr": "Phase {0}"},
    "legend_mean": {"en": "Mean (w/o Outliers)", "pt": "Média (s/ Outliers)", "fr": "Moyenne (sans Aberrants)"},
    "legend_outlier": {"en": "Outliers", "pt": "Outliers", "fr": "Valeurs Aberrantes"},
    "error_read": {"en": "Error processing data: {0}", "pt": "Erro ao processar dados: {0}", "fr": "Erreur de traitement: {0}"},
    "error_cols": {"en": "Column error.", "pt": "Erro nas colunas.", "fr": "Erreur de colonne."},
    "error_proc": {"en": "Error processing data: {0}", "pt": "Erro ao processar dados: {0}", "fr": "Erreur de traitement: {0}"},
    "sidebar_outlier_header": {
        "en": "Outliers & Robustness",
        "pt": "Outliers e Robustez",
        "fr": "Valeurs Aberrantes & Robustesse"
    },
    "outlier_method_label": {
        "en": "Outlier Removal Method",
        "pt": "Método de Remoção de Outliers",
        "fr": "Méthode de Suppression des Valeurs Aberrantes"
    },
    "outlier_none": {
        "en": "No removal (use all points)",
        "pt": "Nenhuma remoção (usar todos os pontos)",
        "fr": "Aucune suppression (utiliser tous les points)"
    },
    "outlier_simple": {
        "en": "ROUT-like (Simple MAD, Z > 2.5)",
        "pt": "ROUT-like (MAD simples, Z > 2,5)",
        "fr": "ROUT-like (MAD simple, Z > 2,5)"
    },
    "outlier_rout": {
        "en": "ROUT (Robust + FDR)",
        "pt": "ROUT (Robusto + FDR)",
        "fr": "ROUT (Robuste + FDR)"
    },
    "rout_q_label": {
        "en": "ROUT Q (Max FDR %)",
        "pt": "ROUT Q (FDR máx. %)",
        "fr": "ROUT Q (FDR max. %)"
    },
    "constraints_header": {
        "en": "Constraints",
        "pt": "Restrições",
        "fr": "Contraintes"
    },
    "force_yi": {
        "en": "Force y_i = 0",
        "pt": "Forçar y_i = 0",
        "fr": "Forcer y_i = 0"
    },
    "force_yf": {
        "en": "Force y_f = 0",
        "pt": "Forçar y_f = 0",
        "fr": "Forcer y_f = 0"
    },
    "summary_header_used": {
        "en": "{0} used",
        "pt": "{0} usado",
        "fr": "{0} utilisé"
    },
    "info_selection_criteria": {
        "en": "Model selection criteria: **{0}** (N = {1}, k_min = {2}, k_max = {3}, N/k_max = {4:.1f}). The selected number of phases is {5} (first local minimum of {0}).",
        "pt": "Critério de seleção de modelo: **{0}** (N = {1}, k_min = {2}, k_max = {3}, N/k_max = {4:.1f}). O número de fases selecionado é {5} (primeiro mínimo local de {0}).",
        "fr": "Critère de sélection du modèle : **{0}** (N = {1}, k_min = {2}, k_max = {3}, N/k_max = {4:.1f}). Le nombre de phases sélectionné est {5} (premier minimum local de {0})."
    },
    "table_col_metric": {"en": "Metric", "pt": "Métrica", "fr": "Métrique"},
    "table_col_value": {"en": "Value", "pt": "Valor", "fr": "Valeur"},
    "table_col_param": {"en": "Param", "pt": "Parâm", "fr": "Param"},
    "table_col_val": {"en": "Val", "pt": "Valor", "fr": "Val"},
    "table_col_se": {"en": "SE", "pt": "EP", "fr": "ET"},
    "table_col_phase": {"en": "Phase", "pt": "Fase", "fr": "Phase"}
}

# Variable Labels Configuration (Using neutral keys)
VAR_LABELS = {
    "generic": {
        "label": {"en": "Generic y(t)", "pt": "Genérico y(t)", "fr": "Générique y(t)"},
        "axis": {"en": "Response (y)", "pt": "Resposta (y)", "fr": "Réponse (y)"},
        "params": ("y_i", "y_f"),
        "rate": "r_max"
    },
    "product": {
        "label": {"en": "Product P(t)", "pt": "Produto P(t)", "fr": "Produit P(t)"},
        "axis": {"en": "Product Conc. (P)", "pt": "Concentração de Produto (P)", "fr": "Concentration en Produit (P)"},
        "params": ("P_i", "P_f"),
        "rate": "r_P,max"
    },
    "substrate": {
        "label": {"en": "Substrate S(t)", "pt": "Substrato S(t)", "fr": "Substrat S(t)"},
        "axis": {"en": "Substrate Conc. (S)", "pt": "Concentração de Substrato (S)", "fr": "Concentration en Substrat (S)"},
        "params": ("S_i", "S_f"),
        "rate": "r_S,max"
    },
    "biomass": {
        "label": {"en": "Biomass X(t)", "pt": "Biomassa X(t)", "fr": "Biomasse X(t)"},
        "axis": {"en": "Biomass Conc. (X)", "pt": "Concentração Celular (X)", "fr": "Concentration Cellulaire (X)"},
        "params": ("X_i", "X_f"),
        "rate": "µ_max"
    }
}

# ==============================================================================
# 1. MATHEMATICAL MODELS
# ==============================================================================

def boltzmann_term_eq31(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Boltzmann model term (Eq. 31)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
    p_safe = max(p_j, 1e-12)
    numerator = 4.0 * r_max_j * (lambda_j - t)
    denominator = delta_y * p_safe
    exponent = (numerator / denominator) + 2.0
    exponent = np.clip(exponent, -500.0, 500.0)
    return p_safe / (1.0 + np.exp(exponent))

def gompertz_term_eq32(t, y_i, y_f, p_j, r_max_j, lambda_j):
    """Gompertz model term (Eq. 32)."""
    delta_y = y_f - y_i
    if abs(delta_y) < 1e-9:
        delta_y = 1e-9
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
    z = theta[2 : 2 + n_phases]
    r_max = theta[2 + n_phases : 2 + 2 * n_phases]
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)
    sum_phases = 0.0
    for j in range(n_phases):
        sum_phases += model_func(t, y_i, y_f, p[j], r_max[j], lambda_[j])
    return y_i + (y_f - y_i) * sum_phases

def sse_loss(theta, t, y, model_func, n_phases):
    """Sum of Squared Errors Loss function."""
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    if np.any(np.diff(lambda_) <= 0):
        return 1e12

    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    return np.sum((y - y_pred) ** 2)

def robust_loss(theta, t, y, model_func, n_phases):
    """Soft L1 robust loss (for ROUT pre-fit)."""
    lambda_ = theta[2 + 2 * n_phases : 2 + 3 * n_phases]
    if np.any(np.diff(lambda_) <= 0):
        return 1e12

    y_pred = polyauxic_model(t, theta, model_func, n_phases)
    if np.any(y_pred < -0.1 * np.max(np.abs(y))):
        return 1e12
    residuals = y - y_pred
    loss = 2.0 * (np.sqrt(1.0 + residuals**2) - 1.0)
    return np.sum(loss)

def numerical_hessian(func, theta, args, epsilon=1e-5):
    """Numerical Hessian calculation."""
    k = len(theta)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i = np.zeros(k)
            e_i[i] = epsilon
            e_j = np.zeros(k)
            e_j[j] = epsilon
            f_pp = func(theta + e_i + e_j, *args)
            f_pm = func(theta + e_i - e_j, *args)
            f_mp = func(theta - e_i + e_j, *args)
            f_mm = func(theta - e_i - e_j, *args)
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
    return hess

def detect_outliers(y_true, y_pred):
    """ROUT-based outlier detection (simple MAD z-score > 2.5)."""
    residuals = y_true - y_pred
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma_robust = 1.4826 * mad if mad > 1e-9 else 1e-9
    z_scores = np.abs(residuals - median_res) / sigma_robust
    return z_scores > 2.5

def detect_outliers_rout_rigorous(y_true, y_pred, Q=1.0):
    """
    ROUT (Rigorous) with FDR control:
    - Robust scale via MAD.
    - t-like scores -> p-values via Student t.
    - Benjamini–Hochberg FDR at Q%.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    n = residuals.size
    if n < 3:
        return np.zeros_like(residuals, dtype=bool)

    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    rsdr = 1.4826 * mad_res if mad_res > 1e-12 else 1e-12

    t_scores = residuals / rsdr
    df = max(n - 1, 1)
    abs_t = np.abs(t_scores)
    p_values = 2.0 * (1.0 - t_dist.cdf(abs_t, df=df))

    alpha = Q / 100.0
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    i = np.arange(1, n + 1)
    bh_thresholds = (i / n) * alpha
    below = p_sorted <= bh_thresholds
    if not np.any(below):
        return np.zeros_like(residuals, dtype=bool)

    k_max = np.max(np.where(below)[0])
    p_crit = p_sorted[k_max]
    mask_outliers = p_values <= p_crit
    return mask_outliers

def smart_initial_guess(t, y, n_phases):
    """Initial parameter guessing based on derivatives."""
    dy = np.gradient(y, t)
    dy_smooth = np.convolve(dy, np.ones(5) / 5, mode='same')
    min_dist = max(1, len(t) // (n_phases * 4))
    peaks, props = find_peaks(dy_smooth, height=np.max(dy_smooth) * 0.1, distance=min_dist)
    guesses = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        best_peaks = peaks[sorted_indices][:n_phases]
        for p_idx in best_peaks:
            guesses.append({'lambda': t[p_idx], 'r_max': dy_smooth[p_idx]})
    while len(guesses) < n_phases:
        t_span = t.max() - t.min()
        guesses.append({
            'lambda': t.min() + t_span * (len(guesses) + 1) / (n_phases + 1),
            'r_max': (np.max(y) - np.min(y)) / (t_span / n_phases)
        })
    guesses.sort(key=lambda x: x['lambda'])
    theta_guess = np.zeros(2 + 3 * n_phases)
    
    # Updated: Detect trend to allow decay (remove implicit yi < yf constraint)
    n_slice = max(1, len(y) // 5) 
    mean_start = np.mean(y[:n_slice])
    mean_end = np.mean(y[-n_slice:])
    
    if float(mean_start) < float(mean_end):
         # Growth trend
         theta_guess[0] = np.min(y)
         theta_guess[1] = np.max(y)
    else:
         # Decay trend (or ambiguous)
         theta_guess[0] = np.max(y)
         theta_guess[1] = np.min(y)

    theta_guess[2 : 2 + n_phases] = 0.0
    for i in range(n_phases):
        theta_guess[2 + n_phases + i] = guesses[i]['r_max']
        theta_guess[2 + 2 * n_phases + i] = guesses[i]['lambda']
    return theta_guess

def calculate_p_errors(z_vals, cov_z):
    """Standard error calculation for p (Softmax)."""
    exps = np.exp(z_vals - np.max(z_vals))
    p = exps / np.sum(exps)
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, j] = p[i] * (1 - p[i])
            else:
                J[i, j] = -p[i] * p[j]
    try:
        cov_p = J @ cov_z @ J.T
        se_p = np.sqrt(np.abs(np.diag(cov_p)))
        return se_p
    except:
        return np.full(n, np.nan)

# ==============================================================================
# 2. FITTING ENGINE
# ==============================================================================

def fit_model_auto(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False):
    """Main fitting function (SSE-based)."""
    SEED_VALUE = 42
    np.random.seed(SEED_VALUE)

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
    
    # Override guesses if forced
    if force_yi:
        theta0_norm[0] = 0.0
    if force_yf:
        theta0_norm[1] = 0.0

    theta0_norm[2 : 2 + n_phases] = 0.0
    theta0_norm[2 + n_phases : 2 + 2 * n_phases] = theta_smart[2 + n_phases : 2 + 2 * n_phases] / (y_scale / t_scale)
    theta0_norm[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_smart[2 + 2 * n_phases : 2 + 3 * n_phases] / t_scale

    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    # Enforce exact zero in population if forced
    if force_yi:
        init_pop[:, 0] = 0.0
    if force_yf:
        init_pop[:, 1] = 0.0

    bounds = []
    
    # y_i bounds
    if force_yi:
        bounds.append((0.0, 1e-10)) # Effectively 0
    else:
        bounds.append((0.0, 1.5)) # CHANGED: Strictly non-negative
    
    # y_f bounds
    if force_yf:
        bounds.append((0.0, 1e-10)) # Effectively 0
    else:
        bounds.append((0.0, 2.0))
        
    for _ in range(n_phases):
        bounds.append((-10, 10))     # z
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))  # r_max_norm
    for _ in range(n_phases):
        # Changed: Ensure lambda >= 0 (and naturally lambda_1 >= 0)
        bounds.append((0.0, 1.2))    # lambda_norm

    res_de = differential_evolution(
        sse_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=SEED_VALUE,
        polish=True,
        tol=1e-6
    )

    res_opt = minimize(
        sse_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-10
    )

    theta_norm = res_opt.x

    theta_real = np.zeros_like(theta_norm)
    se_real = np.zeros_like(theta_norm)
    se_p = np.full(n_phases, np.nan)

    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2 : 2 + n_phases] = theta_norm[2 : 2 + n_phases]
    scale_r = y_scale / t_scale
    theta_real[2 + n_phases : 2 + 2 * n_phases] = theta_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
    scale_l = t_scale
    theta_real[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

    try:
        H_norm = numerical_hessian(sse_loss, theta_norm, args=(t_norm, y_norm, model_func, n_phases))
        y_pred_norm = polyauxic_model(t_norm, theta_norm, model_func, n_phases)
        sse_val_norm = np.sum((y_norm - y_pred_norm) ** 2)
        n_obs = len(y_norm)
        n_p = len(theta_norm)
        sigma2 = sse_val_norm / (n_obs - n_p) if n_obs > n_p else 1e-9
        cov_norm = sigma2 * np.linalg.pinv(H_norm)
        se_norm = np.sqrt(np.abs(np.diag(cov_norm)))
        se_real[0:2] = se_norm[0:2] * scale_y
        se_real[2 : 2 + n_phases] = se_norm[2 : 2 + n_phases]
        se_real[2 + n_phases : 2 + 2 * n_phases] = se_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
        se_real[2 + 2 * n_phases : 2 + 3 * n_phases] = se_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

        idx_z_start = 2
        idx_z_end = 2 + n_phases
        cov_z = cov_norm[idx_z_start:idx_z_end, idx_z_start:idx_z_end]
        z_vals = theta_norm[idx_z_start:idx_z_end]
        se_p = calculate_p_errors(z_vals, cov_z)
    except:
        se_real = np.full_like(theta_real, np.nan)

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    outliers = detect_outliers(y_data, y_pred)

    sse = np.sum((y_data - y_pred) ** 2)
    sst = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - sse / sst
    n_len = len(y_data)
    k = len(theta_real)
    if sse <= 1e-12:
        sse = 1e-12
    if (n_len - k - 1) > 0:
        r2_adj = 1 - (1 - r2) * (n_len - 1) / (n_len - k - 1)
    else:
        r2_adj = np.nan
    aic = n_len * np.log(sse / n_len) + 2 * k
    bic = n_len * np.log(sse / n_len) + k * np.log(n_len)
    aicc = aic + (2 * k * (k + 1)) / (n_len - k - 1) if (n_len - k - 1) > 0 else np.inf

    return {
        "n_phases": n_phases,
        "theta": theta_real,
        "se": se_real,
        "se_p": se_p,
        "metrics": {"R2": r2, "R2_adj": r2_adj, "SSE": sse, "AIC": aic, "BIC": bic, "AICc": aicc},
        "outliers": outliers,
        "y_pred": y_pred
    }

def fit_model_auto_robust_pre(t_data, y_data, model_func, n_phases, force_yi=False, force_yf=False):
    """
    Robust pre-fit (Soft L1) used only to detect outliers (ROUT FDR).
    No SE / information criteria here.
    """
    SEED_VALUE = 42
    np.random.seed(SEED_VALUE)

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
    
    # Override guesses if forced
    if force_yi:
        theta0_norm[0] = 0.0
    if force_yf:
        theta0_norm[1] = 0.0
        
    theta0_norm[2 : 2 + n_phases] = 0.0
    theta0_norm[2 + n_phases : 2 + 2 * n_phases] = theta_smart[2 + n_phases : 2 + 2 * n_phases] / (y_scale / t_scale)
    theta0_norm[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_smart[2 + 2 * n_phases : 2 + 3 * n_phases] / t_scale

    pop_size = 50
    init_pop = np.tile(theta0_norm, (pop_size, 1))
    init_pop *= np.random.uniform(0.8, 1.2, init_pop.shape)

    # Enforce exact zero in population if forced
    if force_yi:
        init_pop[:, 0] = 0.0
    if force_yf:
        init_pop[:, 1] = 0.0

    bounds = []
    
    # y_i bounds
    if force_yi:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 1.5)) # CHANGED: Strictly non-negative
    
    # y_f bounds
    if force_yf:
        bounds.append((0.0, 1e-10))
    else:
        bounds.append((0.0, 2.0))
        
    for _ in range(n_phases):
        bounds.append((-10, 10))     # z
    for _ in range(n_phases):
        bounds.append((0.0, 500.0))  # r_max_norm
    for _ in range(n_phases):
        # Changed: Ensure lambda >= 0 (and naturally lambda_1 >= 0)
        bounds.append((0.0, 1.2))    # lambda_norm

    res_de = differential_evolution(
        robust_loss,
        bounds,
        args=(t_norm, y_norm, model_func, n_phases),
        maxiter=3000,
        popsize=pop_size,
        init=init_pop,
        strategy='best1bin',
        seed=SEED_VALUE,
        polish=True,
        tol=1e-6
    )

    res_opt = minimize(
        robust_loss,
        res_de.x,
        args=(t_norm, y_norm, model_func, n_phases),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-10
    )

    theta_norm = res_opt.x

    theta_real = np.zeros_like(theta_norm)
    scale_y = np.array([y_scale, y_scale])
    theta_real[0:2] = theta_norm[0:2] * scale_y
    theta_real[2 : 2 + n_phases] = theta_norm[2 : 2 + n_phases]
    scale_r = y_scale / t_scale
    theta_real[2 + n_phases : 2 + 2 * n_phases] = theta_norm[2 + n_phases : 2 + 2 * n_phases] * scale_r
    scale_l = t_scale
    theta_real[2 + 2 * n_phases : 2 + 3 * n_phases] = theta_norm[2 + 2 * n_phases : 2 + 3 * n_phases] * scale_l

    y_pred = polyauxic_model(t_data, theta_real, model_func, n_phases)
    return {"theta": theta_real, "y_pred": y_pred}

# ==============================================================================
# 3. DATA PROCESSING
# ==============================================================================

def process_data(df):
    """Processes DataFrame detecting replicates in pairs of columns."""
    # Remove empty columns only if they are entirely empty, but avoid dropping duplicate names issues
    # Better to just iterate based on index
    
    # Ensure index is clean
    df = df.reset_index(drop=True)
    
    num_cols = df.shape[1]
    num_replicates = num_cols // 2
    
    all_t = []
    all_y = []
    replicates = []
    
    for i in range(num_replicates):
        # Use iloc to select columns by position, ensuring we get a Series (1D)
        # We explicitly access .values to bypass any pandas Index ambiguity
        t_col_raw = df.iloc[:, 2 * i].values
        y_col_raw = df.iloc[:, 2 * i + 1].values
        
        # Force conversion to 1D numpy array of floats
        t_vals = pd.to_numeric(t_col_raw, errors='coerce')
        y_vals = pd.to_numeric(y_col_raw, errors='coerce')
        
        # Ensure they are flat numpy arrays (defensive programming)
        if hasattr(t_vals, 'to_numpy'): t_vals = t_vals.to_numpy()
        if hasattr(y_vals, 'to_numpy'): y_vals = y_vals.to_numpy()
            
        t_vals = np.array(t_vals).flatten()
        y_vals = np.array(y_vals).flatten()
        
        mask = ~np.isnan(t_vals) & ~np.isnan(y_vals)
        t_clean = t_vals[mask]
        y_clean = y_vals[mask]
        
        all_t.extend(t_clean)
        all_y.extend(y_clean)
        replicates.append({'t': t_clean, 'y': y_clean, 'name': f'Replica {i + 1}'})
        
    t_flat = np.array(all_t)
    y_flat = np.array(all_y)
    
    # Ensure t_flat is also flat in case extend did something weird (unlikely)
    t_flat = t_flat.flatten()
    y_flat = y_flat.flatten()
    
    if len(t_flat) > 0:
        idx_sort = np.argsort(t_flat)
        return t_flat[idx_sort], y_flat[idx_sort], replicates
    else:
        return np.array([]), np.array([]), []

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
# 4. VISUALIZATION & APP STRUCTURE
# ==============================================================================

def plot_raw_data(replicates, lang):
    """Plots raw data before analysis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for rep in replicates:
        ax.scatter(rep['t'], rep['y'], facecolors='white', edgecolors='black', alpha=0.8, s=20)
    ax.set_title("Experimental Data / Dados Experimentais", fontsize=12)
    ax.set_xlabel(TEXTS['axis_time'][lang])
    ax.set_ylabel("Response (y)")
    ax.grid(True, linestyle=':', alpha=0.3)
    st.pyplot(fig)

def plot_final_summary(replicates, best_results, lang):
    """Plots raw data + best fits for Gompertz and Boltzmann."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. Plot raw data
    # CHANGED: Use enumerate(replicates) to avoid dictionary comparison error
    for i, rep in enumerate(replicates):
        label_txt = 'Data' if i == 0 else ""
        ax.scatter(rep['t'], rep['y'], facecolors='white', edgecolors='black', alpha=0.6, s=20, label=label_txt)
        
    # 2. Determine absolute best model for outliers
    best_aic_val = float('inf')
    best_overall_res = None
    best_overall_func = None
    
    colors = {'Gompertz': 'tab:blue', 'Boltzmann': 'tab:orange'}
    
    # 3. Plot curves
    all_t = [r['t'].max() for r in replicates]
    t_max = max(all_t) if all_t else 1.0
    t_smooth = np.linspace(0, t_max, 300)
    
    for model_name, res in best_results.items():
        if res is None: continue
        func = gompertz_term_eq32 if model_name == "Gompertz" else boltzmann_term_eq31
        y_smooth = polyauxic_model(t_smooth, res['theta'], func, res['n_phases'])
        
        # Check for best overall (using AICc)
        if res['metrics']['AICc'] < best_aic_val:
            best_aic_val = res['metrics']['AICc']
            best_overall_res = res
            best_overall_func = func
            
        label = f"{model_name}: {res['n_phases']} phases (AICc: {res['metrics']['AICc']:.1f})"
        ax.plot(t_smooth, y_smooth, linewidth=2, color=colors.get(model_name, 'black'), label=label)

    # 4. Plot outliers from the absolute best model
    if best_overall_res is not None:
        outlier_count = np.sum(best_overall_res['outliers'])
        ax.set_title(f"Best Fit Summary (Outliers detected by best model: {outlier_count})", fontsize=12)

    ax.set_xlabel(TEXTS['axis_time'][lang])
    ax.set_ylabel("Response")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_metrics_summary(results_list, model_name, lang):
    """Generates a summary chart of metrics vs phases."""
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

    ax2.plot(phases, r2_adj, 'o-', label='Adjusted R²')
    ax2.set_xlabel('Number of Phases')
    ax2.set_ylabel('Adjusted R²')
    ax2.set_title('Fit Quality')
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

    z = theta[2 : 2 + n]
    r_max = theta[2 + n : 2 + 2 * n]
    r_max_se = se[2 + n : 2 + 2 * n]
    lambda_ = theta[2 + 2 * n : 2 + 3 * n]
    lambda_se = se[2 + 2 * n : 2 + 3 * n]
    p = np.exp(z - np.max(z))
    p /= np.sum(p)

    phases = []
    for i in range(n):
        phases.append({
            "p": p[i],
            "SE p": se_p[i],
            "r_max": r_max[i],
            "r_max_se": r_max_se[i],
            "lambda": lambda_[i],
            "lambda_se": lambda_se[i]
        })
    phases.sort(key=lambda x: x['lambda'])

    c_plot, c_data = st.columns([1.5, 1])
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Replicates: white markers with black edge
        for rep in replicates:
            ax.scatter(
                rep['t'],
                rep['y'],
                facecolors='white',
                edgecolors='black',
                alpha=0.8,
                s=15,
                marker='o'
            )

        outliers = raw_data_w_outliers[raw_data_w_outliers['is_outlier']]
        if not outliers.empty:
            ax.scatter(
                outliers['t'],
                outliers['y'],
                color='red',
                marker='x',
                s=50,
                label=TEXTS['legend_outlier'][lang],
                zorder=5
            )

        # Mean + error bar only if multiple replicates
        if len(replicates) > 1:
            ax.errorbar(
                stats_df['t_round'],
                stats_df['mean'],
                yerr=stats_df['std'],
                fmt='o',
                color='black',
                ecolor='black',
                capsize=3,
                label=TEXTS['legend_mean'][lang],
                zorder=4
            )

        t_max_val = raw_data_w_outliers['t'].max()
        t_smooth = np.linspace(0, t_max_val, 300)
        y_smooth = polyauxic_model(t_smooth, theta, model_func, n)
        ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label=TEXTS['legend_global'][lang])

        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i, ph in enumerate(phases):
            y_ind = model_func(t_smooth, y_i, y_f, ph['p'], ph['r_max'], ph['lambda'])
            y_vis = y_i + (y_f - y_i) * y_ind
            ax.plot(
                t_smooth,
                y_vis,
                '--',
                color=colors[i],
                alpha=0.6,
                label=TEXTS['legend_phase'][lang].format(i + 1)
            )

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
        # Translated DataFrame for Globals
        df_glob = pd.DataFrame(
            {
                TEXTS['table_col_param'][lang]: [yi_name, yf_name], 
                TEXTS['table_col_val'][lang]: [y_i, y_f], 
                TEXTS['table_col_se'][lang]: [y_i_se, y_f_se]
            }
        )
        st.dataframe(df_glob.style.format({TEXTS['table_col_val'][lang]: "{:.4f}", TEXTS['table_col_se'][lang]: "{:.4f}"}), hide_index=True)

        rows = []
        for i, ph in enumerate(phases):
            rows.append({
                TEXTS['table_col_phase'][lang]: i + 1,
                "p": ph['p'],
                f"{TEXTS['table_col_se'][lang]} p": ph['SE p'],
                rate_label: ph['r_max'],
                f"{TEXTS['table_col_se'][lang]} {rate_label}": ph['r_max_se'],
                "λ": ph['lambda'],
                f"{TEXTS['table_col_se'][lang]} λ": ph['lambda_se']
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({
                "p": "{:.4f}",
                f"{TEXTS['table_col_se'][lang]} p": "{:.4f}",
                rate_label: "{:.4f}",
                f"{TEXTS['table_col_se'][lang]} {rate_label}": "{:.4f}",
                "λ": "{:.4f}",
                f"{TEXTS['table_col_se'][lang]} λ": "{:.4f}"
            }),
            hide_index=True
        )

        m = res['metrics']
        df_met = pd.DataFrame(
            {
                TEXTS['table_col_metric'][lang]: ["R²", "R² Adj", "AIC", "AICc", "BIC"],
                TEXTS['table_col_value'][lang]: [m['R2'], m['R2_adj'], m['AIC'], m['AICc'], m['BIC']]
            }
        )
        st.dataframe(df_met.style.format({TEXTS['table_col_value'][lang]: "{:.4f}"}), hide_index=True)

# ==============================================================================
# 5. INFORMATION CRITERIA SELECTION HELPERS
# ==============================================================================

def choose_information_criterion(N, k_max):
    """
    Escolhe AIC, AICc ou BIC com base em N e k, conforme Tabela 1 do artigo.
    - N <= 200:
        - N/k < 40 -> AICc (Small sample correction)
        - N/k >= 40 -> AIC
    - N > 200 e k grande -> BIC (Parsimony).
    """
    dof_ratio = N / max(k_max, 1)
    if N <= 200:
        if dof_ratio < 40:
            return "AICc"
        else:
            return "AIC"
    else:
        # For large N, if k is large, BIC is safer. If k is small, BIC is also fine.
        return "BIC"

def select_first_local_min_index(values, tol=1e-9):
    """
    Retorna o índice do primeiro mínimo local (no sentido do artigo):
    varre na ordem das fases; enquanto o critério melhora (diminui), segue.
    Na primeira vez que o valor deixa de melhorar (fica igual ou aumenta),
    retorna o índice do melhor até aquele ponto.
    Exemplo: [100, 50, 75, 10] -> índice 1 (2 fases).
    """
    if not values:
        return 0
    best_idx = 0
    for i in range(1, len(values)):
        if values[i] < values[best_idx] - tol:
            best_idx = i
        elif values[i] >= values[best_idx] - tol:
            # parou de melhorar (igual ou pior) -> retorna mínimo anterior
            break
    return best_idx

# ==============================================================================
# 6. MAIN APP
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Analysis")

    # Sidebar for settings
    st.sidebar.header("Language / Idioma / Langue")
    lang_key = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_key]

    st.title(TEXTS['app_title'][lang])

    # Intro and Instructions
    st.info(TEXTS['intro_desc'][lang])

    # References with Badges (Flexbox Layout)
    st.markdown(f"**{TEXTS['paper_ref'][lang]}**")

    # Variáveis do Zenodo
    zenodo_doi = "10.5281/zenodo.18025828"
    zenodo_url = f"https://doi.org/{zenodo_doi}"
    # Badge estilo Shields.io para combinar com o do ArXiv
    zenodo_badge_img = f"https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18025828-blue.svg?logo=zenodo&logoColor=white"

    badge_html = f"""
    <div style="display: flex; flex-direction: column; gap: 12px;">
        
        <div style="display: flex; align-items: center; gap: 15px;">
            <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-arxiv-id='2507.05960' data-hide-no-mentions='true'></div>
            <div style="font-family: 'Times New Roman', serif; font-size: 16px;">
                Mockaitis, G. (2025) Mono and Polyauxic Growth Kinetic Models. ArXiv: 2507.05960, 24 p.
            </div>
            <a href="https://doi.org/10.48550/arXiv.2507.05960" target="_blank">
                <img src="https://img.shields.io/badge/arXiv-2507.05960-b31b1b.svg" alt="arXiv">
            </a>
            <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub">
            </a>
        </div>

        <div style="display: flex; align-items: center; gap: 15px;">
            <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-doi='{zenodo_doi}'></div>
            
            <div style="font-family: 'Times New Roman', serif; font-size: 16px;">
                {TEXTS['zenodo_cite'][lang]}
            </div>
            
            <a href="{zenodo_url}" target="_blank">
                <img src="{zenodo_badge_img}" alt="Zenodo DOI">
            </a>
        </div>

        <script type='text/javascript' src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>
    </div>
    """
    
    components.html(badge_html, height=150)

    with st.expander(TEXTS['instructions_header'][lang], expanded=False):
        st.markdown(TEXTS['instructions_list'][lang])
    st.markdown("---")

    # Main Analysis Interface
    st.sidebar.header(TEXTS['sidebar_config'][lang])

    var_type_opts = list(VAR_LABELS.keys())

    selected_var_key = st.sidebar.selectbox(
        TEXTS['var_type'][lang],
        options=var_type_opts,
        format_func=lambda x: VAR_LABELS[x]['label'][lang]
    )

    config = VAR_LABELS[selected_var_key]
    y_label = config['axis'][lang]
    param_labels = config['params']
    rate_label = config['rate']

    file = st.sidebar.file_uploader(TEXTS['upload_label'][lang], type=["csv", "xlsx"])
    max_phases = st.sidebar.number_input(TEXTS['max_phases'][lang], 1, 10, 5)

    # --- NEW: Outlier handling configuration (trilingual) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['sidebar_outlier_header'][lang]}")
    outlier_options_keys = ["none", "simple", "rout"]
    outlier_method_key = st.sidebar.selectbox(
        TEXTS['outlier_method_label'][lang],
        options=outlier_options_keys,
        format_func=lambda k: TEXTS[f"outlier_{k}"][lang]
    )
    rout_q = 1.0
    if outlier_method_key == "rout":
        rout_q = st.sidebar.slider(
            TEXTS["rout_q_label"][lang],
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    
    # --- NEW: Constraints ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['constraints_header'][lang]}")
    
    # Force y_i = 0
    force_yi = st.sidebar.checkbox(TEXTS['force_yi'][lang], value=False)
    
    # Force y_f = 0 (disabled if y_i is active)
    force_yf = st.sidebar.checkbox(TEXTS['force_yf'][lang], value=False, disabled=force_yi)
    
    # Logic safety: if yi is forced, yf cannot be True
    if force_yi:
        force_yf = False

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            t_flat, y_flat, replicates = process_data(df)
            if not replicates:
                st.error(TEXTS['error_cols'][lang])
            else:
                # --- NEW: Placeholder for Top Graph ---
                graph_placeholder = st.empty()
                with graph_placeholder:
                    plot_raw_data(replicates, lang)
                # -------------------------------------

                st.success(TEXTS['data_loaded'][lang].format(len(replicates), len(t_flat)))
                if st.button(TEXTS['run_button'][lang]):
                    st.divider()
                    
                    # Store best results globally for the top graph update
                    best_results_global = {"Gompertz": None, "Boltzmann": None}
                    
                    tab1, tab2 = st.tabs(
                        [TEXTS['tab_gompertz'][lang], TEXTS['tab_boltzmann'][lang]]
                    )
                    for tab, model_name, func, color in [
                        (tab1, "Gompertz", gompertz_term_eq32, "tab:blue"),
                        (tab2, "Boltzmann", boltzmann_term_eq31, "tab:orange")
                    ]:
                        with tab:
                            results_list = []
                            for n in range(1, max_phases + 1):
                                with st.expander(
                                    TEXTS['expanding'][lang].format(model_name, n),
                                    expanded=False
                                ):
                                    with st.spinner(TEXTS['optimizing'][lang].format(n)):

                                        res = None

                                        # --- Outlier pipeline applied to loaded data ---
                                        if outlier_method_key == "none":
                                            res = fit_model_auto(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)

                                        elif outlier_method_key == "simple":
                                            # 1) Pre-fit with SSE
                                            res_pre = fit_model_auto(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)
                                            if res_pre:
                                                y_pred_pre = res_pre["y_pred"]
                                                mask = detect_outliers(y_flat, y_pred_pre)
                                                n_params = 2 + 3 * n
                                                # Require some margin of points after removal
                                                if np.any(mask) and (len(y_flat[~mask]) > n_params + 5):
                                                    t_clean = t_flat[~mask]
                                                    y_clean = y_flat[~mask]
                                                    res = fit_model_auto(t_clean, y_clean, func, n, force_yi=force_yi, force_yf=force_yf)
                                                else:
                                                    res = res_pre

                                        elif outlier_method_key == "rout":
                                            # 1) Robust pre-fit
                                            res_robust = fit_model_auto_robust_pre(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)
                                            if res_robust:
                                                y_pred_pre = res_robust["y_pred"]
                                                mask = detect_outliers_rout_rigorous(y_flat, y_pred_pre, Q=rout_q)
                                                n_params = 2 + 3 * n
                                                if np.any(mask) and (len(y_flat[~mask]) > n_params + 5):
                                                    t_clean = t_flat[~mask]
                                                    y_clean = y_flat[~mask]
                                                    res = fit_model_auto(t_clean, y_clean, func, n, force_yi=force_yi, force_yf=force_yf)
                                                else:
                                                    res = fit_model_auto(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)

                                        # ------------------------------------------------

                                        if res:
                                            display_single_fit(
                                                res,
                                                replicates,
                                                func,
                                                color,
                                                y_label,
                                                param_labels,
                                                rate_label,
                                                lang
                                            )
                                            results_list.append(res)
                                        else:
                                            st.warning(TEXTS['warning_insufficient'][lang])

                            if results_list:
                                st.markdown(f"### {TEXTS['table_title'][lang]}")

                                # N e k_max para escolha do critério (Tabela 1)
                                N = len(y_flat)
                                k_values = [len(r['theta']) for r in results_list]
                                k_min, k_max = min(k_values), max(k_values)
                                ic_name = choose_information_criterion(N, k_max)

                                ic_values = [r['metrics'][ic_name] for r in results_list]

                                # Primeiro mínimo local do critério escolhido
                                best_idx = select_first_local_min_index(ic_values)
                                best_n = results_list[best_idx]['n_phases']
                                
                                # Store for top graph update
                                best_results_global[model_name] = results_list[best_idx]

                                summary_data = []
                                for i, r in enumerate(results_list):
                                    m = r['metrics']
                                    summary_data.append({
                                        "F": r['n_phases'],
                                        "R²": m['R2'],
                                        "R² Adj": m['R2_adj'],
                                        "SSE": m['SSE'],
                                        "AIC": m['AIC'],
                                        "AICc": m['AICc'],
                                        "BIC": m['BIC'],
                                        TEXTS['summary_header_used'][lang].format(ic_name): ic_values[i]
                                    })

                                summary_df = pd.DataFrame(summary_data)

                                def highlight_row(row):
                                    if row['F'] == best_n:
                                        return ['background-color: #d4edda; font-weight: bold'] * len(row)
                                    return [''] * len(row)

                                st.dataframe(
                                    summary_df.style.apply(highlight_row, axis=1).format({
                                        "R²": "{:.4f}",
                                        "R² Adj": "{:.4f}",
                                        "SSE": "{:.4f}",
                                        "AIC": "{:.4f}",
                                        "AICc": "{:.4f}",
                                        "BIC": "{:.4f}",
                                        TEXTS['summary_header_used'][lang].format(ic_name): "{:.4f}"
                                    }),
                                    hide_index=True
                                )

                                # Mensagem clara sobre o critério e graus de liberdade
                                ratio = N / k_max
                                st.info(
                                    TEXTS['info_selection_criteria'][lang].format(ic_name, N, k_min, k_max, ratio, best_n)
                                )

                                st.success(
                                    TEXTS['best_model_msg'][lang].format(best_n, ic_name)
                                )

                                st.markdown(f"### {TEXTS['graph_summary_title'][lang]}")
                                plot_metrics_summary(results_list, model_name, lang)

                    # --- UPDATE TOP GRAPH WITH RESULTS ---
                    with graph_placeholder:
                        plot_final_summary(replicates, best_results_global, lang)
                    # -------------------------------------

        except Exception as e:
            st.error(TEXTS['error_proc'][lang].format(e))
    else:
        st.info(TEXTS['info_upload'][lang])

if __name__ == "__main__":
    main()


import streamlit.components.v1 as components

# ==============================================================================
# 7. FOOTER
# ==============================================================================

profile_pic_url = "https://github.com/gusmock.png"
st.markdown("---")

footer_html = f"""
<style>
    /* Main Footer Container */
    .footer-container {{
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #444;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px 0;
    }}
    
    /* Photo and Text Area */
    .profile-section {{
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
        max-width: 800px;
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 600px) {{
        .profile-section {{
            flex-direction: column;
        }}
    }}

    .profile-img {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .profile-info {{
        text-align: left;
    }}
    
    @media (max-width: 600px) {{
        .profile-info {{ text-align: center; }}
    }}

    .profile-info h2 {{
        margin: 0;
        font-size: 16px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .profile-info h4 {{
        margin: 5px 0;
        font-size: 18px;
        color: #222;
        font-weight: 700;
    }}
    
    .profile-info p {{
        margin: 0;
        font-size: 13px;
        color: #666;
        line-height: 1.4;
    }}

    /* Personal Badges Container */
    .social-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
    }}
    
    .social-badges a img {{
        height: 26px;
        border-radius: 4px;
        transition: transform 0.2s, opacity 0.2s;
    }}
    
    .social-badges a img:hover {{
        transform: translateY(-2px);
        opacity: 0.9;
    }}
</style>

<div class="footer-container">
    
    <div class="profile-section">
        <img src="{profile_pic_url}" class="profile-img" alt="Gustavo Mockaitis">
        
        <div class="profile-info">
            <h2>GBMA / FEAGRI / UNICAMP</h2>
            <h4>Dev: Prof. Dr. Gustavo Mockaitis</h4>
            <p>
                Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and Environment<br>
                School of Agricultural Engineering, University of Campinas.<br>
                Campinas, SP, Brazil.
            </p>
        </div>
    </div>

    <div class="social-badges">
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank">
            <img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID">
        </a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=en&oi=ao" target="_blank">
            <img src="https://img.shields.io/badge/Scholar-Profile-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar">
        </a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/ResearchGate-Profile-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate">
        </a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank">
            <img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes CV">
        </a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank">
            <img src="https://img.shields.io/badge/Web_of_Science-Profile-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science">
        </a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/UNICAMP-Institutional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP">
        </a>
    </div>

</div>
"""

components.html(footer_html, height=280, scrolling=False)
