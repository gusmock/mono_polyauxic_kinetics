import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import hashlib
import socket
from datetime import datetime
from streamlit.web.server.websocket_headers import _get_websocket_headers
import uuid

# Import Core Logic from your separated file
from polyauxic_core import (
    boltzmann_term_eq31,
    gompertz_term_eq32,
    polyauxic_model,
    fit_model_auto,
    fit_model_auto_robust_pre,
    process_data,
    calculate_mean_with_outliers,
    choose_information_criterion,
    select_first_local_min_index,
    detect_outliers,
    detect_outliers_rout_rigorous
)

# ==============================================================================
# LOGGING & DATA STORAGE UTILS
# ==============================================================================

def get_user_identifier():
    """
    Attempts to retrieve a unique user identifier (IP address). 
    If it fails or detects localhost, it generates a persistent unique ID for the session.
    """
    # 1. Return persistent ID if already defined in the current session
    if "user_id_persistent" in st.session_state:
        return st.session_state["user_id_persistent"]

    detected_id = None

    # 2. Try to obtain the real IP via headers (useful for Streamlit Cloud/Docker)
    try:
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            ip = headers["X-Forwarded-For"].split(",")[0].strip()
            if ip and ip != "127.0.0.1":
                detected_id = ip
    except Exception:
        pass

    # 3. Fallback: try local hostname
    if not detected_id:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and ip != "127.0.0.1":
                detected_id = ip
        except Exception:
            pass

    # 4. Final Fallback: Generate a short UUID for anonymity and localhost
    if not detected_id or detected_id == "127.0.0.1":
        detected_id = f"anon_{str(uuid.uuid4())[:8]}"

    # 5. Save to session_state to ensure persistence during app usage
    st.session_state["user_id_persistent"] = detected_id
    
    return detected_id

def save_uploaded_data(df):
    """
    Saves the uploaded DataFrame as a CSV in the local /data folder.
    Format: DD-MM-YYYY_IDENTIFIER.csv
    Includes robust try/except blocks to prevent UI crashes if disk writing fails.
    """
    data_dir = "data"
    
    try:
        # Create directory if it does not exist (Safe for local environments)
        os.makedirs(data_dir, exist_ok=True)
    except Exception as e:
        st.warning(f"Could not create local data directory. Backups will be skipped. Error: {e}")
        return
    
    try:
        # 1. Generate MD5 hash of the current content to prevent exact duplicates
        content_csv = df.to_csv(index=False)
        current_hash = hashlib.md5(content_csv.encode('utf-8')).hexdigest()
        
        # 2. Define target filename with the sanitized User Identifier
        date_str = datetime.now().strftime("%d-%m-%Y")
        user_id = get_user_identifier() 
        safe_id = user_id.replace(":", "_").replace(".", "_")
        target_filename = f"{date_str}_{safe_id}.csv"
        target_path = os.path.join(data_dir, target_filename)
        
        # 3. Scan directory to check for exact content duplicates
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    existing_hash = hashlib.md5(existing_content.encode('utf-8')).hexdigest()
                    
                    # If content matches exactly, remove the old file to overwrite it
                    if existing_hash == current_hash and filename != target_filename:
                        os.remove(file_path)
                except Exception as e:
                    # Silent pass to not disrupt the user experience
                    continue

        # 4. Save the new/updated file
        df.to_csv(target_path, index=False)
    except Exception as e:
        st.warning(f"Failed to save local backup file. Process will continue. Error: {e}")

# ==============================================================================
# EXCEL EXPORT UTILITY
# ==============================================================================

def generate_excel_report(best_results_global, replicates, param_labels, rate_label):
    """
    Generates an Excel file (.xlsx) in memory containing all parameters, 
    standard errors, and statistics for the best models found.
    """
    output = io.BytesIO()
    
    # Use xlsxwriter engine to format Excel in memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # Loop through the best results dictionary (Gompertz, Boltzmann)
        for model_name, res in best_results_global.items():
            if res is None:
                continue
                
            n = res['n_phases']
            theta = res['theta']
            se = res['se']
            se_p = res['se_p']
            yi_name, yf_name = param_labels
            
            # --- Global Parameters Sheet ---
            global_params = pd.DataFrame({
                "Parameter": [yi_name, yf_name, "Phases (n)"],
                "Value": [theta[0], theta[1], n],
                "Standard Error (SE)": [se[0], se[1], "N/A"]
            })
            global_params.to_excel(writer, sheet_name=f"{model_name}_Global", index=False)
            
            # --- Phase Specific Parameters Sheet ---
            z = theta[2 : 2 + n]
            r_max = theta[2 + n : 2 + 2 * n]
            r_max_se = se[2 + n : 2 + 2 * n]
            lambda_ = theta[2 + 2 * n : 2 + 3 * n]
            lambda_se = se[2 + 2 * n : 2 + 3 * n]
            
            # Calculate proportion 'p' based on z transformation
            p = np.exp(z - np.max(z))
            p /= np.sum(p)
            
            phase_data = []
            for i in range(n):
                phase_data.append({
                    "Phase": i + 1,
                    "Proportion (p)": p[i],
                    "SE (p)": se_p[i],
                    f"Max Rate ({rate_label})": r_max[i],
                    f"SE ({rate_label})": r_max_se[i],
                    "Lag Phase (lambda)": lambda_[i],
                    "SE (lambda)": lambda_se[i]
                })
            
            phase_df = pd.DataFrame(phase_data)
            # Sort phases sequentially by lambda (lag time)
            phase_df = phase_df.sort_values(by="Lag Phase (lambda)").reset_index(drop=True)
            phase_df.to_excel(writer, sheet_name=f"{model_name}_Phases", index=False)
            
            # --- Statistics & Metrics Sheet ---
            m = res['metrics']
            metrics_df = pd.DataFrame({
                "Metric": ["R-squared", "Adjusted R-squared", "SSE", "AIC", "AICc", "BIC"],
                "Value": [m['R2'], m['R2_adj'], m['SSE'], m['AIC'], m['AICc'], m['BIC']]
            })
            metrics_df.to_excel(writer, sheet_name=f"{model_name}_Metrics", index=False)

    return output.getvalue()


# ==============================================================================
# 0. CONFIGURATION & GLOBAL SETTINGS
# ==============================================================================

# Global Plot Style - Academic Standard (Times New Roman 11)
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

# UI Text Dictionary (Truncated here for brevity, keeping all original definitions)
TEXTS = {
    "app_title": {
        "en": "Polyauxic Modeling Platform",
        "pt": "Plataforma de Modelagem Poliauxica",
        "fr": "Plateforme de Modélisation Polyauxique"
    },
    "intro_desc": {
        "en": "This application performs advanced non-linear regression for microbial growth kinetics. It identifies mono- and polyauxic behaviors using robust statistical methods (Charbonnier loss, ROUT outlier detection) and selects models via Information Criteria (AIC, AICc, BIC).",
        "pt": "Este aplicativo realiza regressão não-linear avançada para cinética microbiana. Identifica comportamentos mono e poliauxicos usando métodos estatísticos robustos (Charbonnier, outliers ROUT) e seleciona modelos via Critérios de Informação (AIC, AICc, BIC).",
        "fr": "Cette application effectue une régression non linéaire avancée pour la cinétique microbienne. Elle identifie les comportements mono- et polyauxiques à l'aide de méthodes robustes (Charbonnier, ROUT) et sélectionne les modèles via Critères d'Information (AIC, AICc, BIC)."
    },
    "paper_ref": {
        "en": "Reference Paper & Source:",
        "pt": "Artigo de Referência e Fonte:",
        "fr": "Article de Référence et Source :"
    },
    "zenodo_cite": {
        "en": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "pt": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "fr": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Logiciel]. Zenodo."
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
        "fr": "Veuillez télécharger un fichier para commencer."
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
    "download_excel": {"en": "Download Full Results (.xlsx)", "pt": "Baixar Resultados Completos (.xlsx)", "fr": "Télécharger les Résultats Complets (.xlsx)"},
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

# Variable Labels Configuration
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
# 4. VISUALIZATION & APP STRUCTURE
# ==============================================================================

def plot_raw_data(replicates, lang):
    """Plots raw data before analysis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for rep in replicates:
        ax.scatter(rep['t'], rep['y'], facecolors='white', edgecolors='black', alpha=0.8, s=20)
    ax.set_title("Experimental Data", fontsize=12)
    ax.set_xlabel(TEXTS['axis_time'][lang])
    ax.set_ylabel("Response (y)")
    ax.grid(True, linestyle=':', alpha=0.3)
    st.pyplot(fig)

def plot_final_summary(replicates, best_results, lang):
    """Plots raw data + best fits for Gompertz and Boltzmann."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i, rep in enumerate(replicates):
        label_txt = 'Data' if i == 0 else ""
        ax.scatter(rep['t'], rep['y'], facecolors='white', edgecolors='black', alpha=0.6, s=20, label=label_txt)
        
    best_aic_val = float('inf')
    best_overall_res = None
    
    colors = {'Gompertz': 'tab:blue', 'Boltzmann': 'tab:orange'}
    
    all_t = [r['t'].max() for r in replicates]
    t_max = max(all_t) if all_t else 1.0
    t_smooth = np.linspace(0, t_max, 300)
    
    for model_name, res in best_results.items():
        if res is None: continue
        func = gompertz_term_eq32 if model_name == "Gompertz" else boltzmann_term_eq31
        y_smooth = polyauxic_model(t_smooth, res['theta'], func, res['n_phases'])
        
        if res['metrics']['AICc'] < best_aic_val:
            best_aic_val = res['metrics']['AICc']
            best_overall_res = res
            
        label = f"{model_name}: {res['n_phases']} phases (AICc: {res['metrics']['AICc']:.1f})"
        ax.plot(t_smooth, y_smooth, linewidth=2, color=colors.get(model_name, 'black'), label=label)

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
    """Displays detailed results for a single fit."""
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
# 6. MAIN APP
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Polyauxic Analysis")

    # --- SESSION STATE INITIALIZATION ---
    # This prevents the app from losing context when download buttons trigger a page rerun.
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'uploaded_file_hash' not in st.session_state:
        st.session_state.uploaded_file_hash = None

    # Sidebar for settings
    st.sidebar.header("Language / Idioma / Langue")
    lang_key = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_key]

    st.title(TEXTS['app_title'][lang])

    # Intro and Instructions
    st.info(TEXTS['intro_desc'][lang])

    # --- REFERENCES SECTION WITH FULL METRICS SUITE ---
    ref_header_text = TEXTS['paper_ref'][lang]
    zenodo_doi = "10.5281/zenodo.18025828"
    zenodo_url = f"https://doi.org/{zenodo_doi}"
    zenodo_badge_img = "https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18025828-blue.svg?logo=zenodo&logoColor=white"
    arxiv_doi = "10.48550/arXiv.2507.05960"
    
    badge_col_width = "210px"
    badge_min_height = "55px"

    badge_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: "Source Sans Pro", sans-serif; margin: 0; padding: 0; color: rgb(49, 51, 63); overflow: visible; }}
            .ref-header {{ font-size: 18px; font-weight: 700; margin-bottom: 15px; }}
            .rows-container {{ display: flex; flex-direction: column; gap: 15px; }}
            .row {{ display: flex; align-items: center; gap: 15px; }}
            .badge-wrapper {{ display: flex; align-items: center; gap: 8px; min-width: {badge_col_width}; min-height: {badge_min_height}; }}
            .citation-text {{ font-family: 'Times New Roman', serif; font-size: 16px; line-height: 1.4; }}
        </style>
    </head>
    <body>
        <div class="ref-header">{ref_header_text}</div>
        <div class="rows-container">
            <div class="row">
                <div class="badge-wrapper">
                    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-arxiv-id='2507.05960' data-hide-no-mentions='true'></div>
                    <a href="https://plu.mx/plum/a/?arxiv=2507.05960" class="plumx-plum-print-popup" data-popup="right" data-size="medium" data-pass-hidden-categories="true"></a>
                    <span class="__dimensions_badge_embed__" data-doi="{arxiv_doi}" data-style="small_circle" data-hide-zero-citations="false"></span>
                </div>
                <div class="citation-text">Mockaitis, G. (2025) Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics. ArXiv: 2507.05960, 42 p.</div>
                <a href="https://doi.org/10.48550/arXiv.2507.05960" target="_blank"><img src="https://img.shields.io/badge/arXiv-2507.05960-b31b1b.svg" alt="arXiv"></a>
                <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
            </div>
            <div class="row">
                <div class="badge-wrapper">
                    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-doi='{zenodo_doi}' data-hide-no-mentions='false'></div>
                    <a href="https://plu.mx/plum/a/?doi={zenodo_doi}" class="plumx-plum-print-popup" data-popup="right" data-size="medium" data-pass-hidden-categories="true"></a>
                </div>
                <div class="citation-text">{TEXTS['zenodo_cite'][lang]}</div>
                <a href="{zenodo_url}" target="_blank"><img src="{zenodo_badge_img}" alt="Zenodo DOI"></a>
            </div>
        </div>
        <script type='text/javascript' src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>
        <script type="text/javascript" src="//cdn.plu.mx/widget-popup.js"></script>
        <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8"></script>
    </body>
    </html>
    """
    
    components.html(badge_html, height=210)
    
    with st.expander(TEXTS['instructions_header'][lang], expanded=False):
        st.markdown(TEXTS['instructions_list'][lang])
    st.markdown("---")

    # Main Analysis Interface Sidebar
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

    # Reset analysis state if a new file is uploaded
    if file is not None:
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if st.session_state.uploaded_file_hash != file_hash:
            st.session_state.analysis_run = False
            st.session_state.uploaded_file_hash = file_hash

    # --- Outlier handling configuration ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['sidebar_outlier_header'][lang]}")
    outlier_options_keys = ["none", "simple", "rout"]
    
    # Store settings changes to reset analysis automatically if parameter changes
    def reset_analysis():
        st.session_state.analysis_run = False

    outlier_method_key = st.sidebar.selectbox(
        TEXTS['outlier_method_label'][lang],
        options=outlier_options_keys,
        format_func=lambda k: TEXTS[f"outlier_{k}"][lang],
        on_change=reset_analysis
    )
    rout_q = 1.0
    if outlier_method_key == "rout":
        rout_q = st.sidebar.slider(
            TEXTS["rout_q_label"][lang],
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            on_change=reset_analysis
        )
    
    # --- Constraints ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['constraints_header'][lang]}")
    
    force_yi = st.sidebar.checkbox(TEXTS['force_yi'][lang], value=False, on_change=reset_analysis)
    force_yf = st.sidebar.checkbox(TEXTS['force_yf'][lang], value=False, disabled=force_yi, on_change=reset_analysis)
    
    if force_yi:
        force_yf = False

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            save_uploaded_data(df)

            t_flat, y_flat, replicates = process_data(df)
            if not replicates:
                st.error(TEXTS['error_cols'][lang])
            else:
                # Always show Raw Data top graph
                plot_raw_data(replicates, lang)

                st.success(TEXTS['data_loaded'][lang].format(len(replicates), len(t_flat)))
                
                # Update Session State based on Button Click
                if st.button(TEXTS['run_button'][lang]):
                    st.session_state.analysis_run = True

                # ==========================================================
                # ANALYSIS EXECUTION BLOCK (Managed by Session State)
                # ==========================================================
                if st.session_state.analysis_run:
                    st.divider()
                    
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

                                        # --- Outlier pipeline ---
                                        if outlier_method_key == "none":
                                            res = fit_model_auto(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)

                                        elif outlier_method_key == "simple":
                                            res_pre = fit_model_auto(t_flat, y_flat, func, n, force_yi=force_yi, force_yf=force_yf)
                                            if res_pre:
                                                y_pred_pre = res_pre["y_pred"]
                                                mask = detect_outliers(y_flat, y_pred_pre)
                                                n_params = 2 + 3 * n
                                                if np.any(mask) and (len(y_flat[~mask]) > n_params + 5):
                                                    t_clean = t_flat[~mask]
                                                    y_clean = y_flat[~mask]
                                                    res = fit_model_auto(t_clean, y_clean, func, n, force_yi=force_yi, force_yf=force_yf)
                                                else:
                                                    res = res_pre

                                        elif outlier_method_key == "rout":
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
                                                res, replicates, func, color, y_label, param_labels, rate_label, lang
                                            )
                                            results_list.append(res)
                                        else:
                                            st.warning(TEXTS['warning_insufficient'][lang])

                            if results_list:
                                st.markdown(f"### {TEXTS['table_title'][lang]}")

                                N = len(y_flat)
                                k_values = [len(r['theta']) for r in results_list]
                                k_min, k_max = min(k_values), max(k_values)
                                ic_name = choose_information_criterion(N, k_max)

                                ic_values = [r['metrics'][ic_name] for r in results_list]
                                best_idx = select_first_local_min_index(ic_values)
                                best_n = results_list[best_idx]['n_phases']
                                
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

                                st.info(
                                    TEXTS['info_selection_criteria'][lang].format(ic_name, N, k_min, k_max, N / k_max, best_n)
                                )

                                st.success(
                                    TEXTS['best_model_msg'][lang].format(best_n, ic_name)
                                )

                                st.markdown(f"### {TEXTS['graph_summary_title'][lang]}")
                                plot_metrics_summary(results_list, model_name, lang)

                    # --- FINAL SUMMARY GRAPH (Appears after tabs) ---
                    st.divider()
                    plot_final_summary(replicates, best_results_global, lang)
                    
                    # --- EXCEL EXPORT BUTTON ---
                    # Placed at the very end of the analysis so users can grab everything at once
                    st.divider()
                    excel_data = generate_excel_report(best_results_global, replicates, param_labels, rate_label)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label=f"📊 {TEXTS['download_excel'][lang]}",
                            data=excel_data,
                            file_name=f"polyauxic_results_{datetime.now().strftime('%d-%m-%Y')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(TEXTS['error_proc'][lang].format(e))
    else:
        st.info(TEXTS['info_upload'][lang])

if __name__ == "__main__":
    main()

# ==============================================================================
# 7. FOOTER
# ==============================================================================

profile_pic_url = "https://github.com/gusmock.png"
st.markdown("---")

footer_css = """
<style>
    /* Main Footer Container */
    .footer-container {
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #444;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Photo and Text Area */
    .profile-section {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
        max-width: 800px;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .profile-section { flex-direction: column; }
    }

    .profile-img {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .profile-info { text-align: left; }
    
    @media (max-width: 600px) {
        .profile-info { text-align: center; }
    }

    .profile-info h2 {
        margin: 0;
        font-size: 16px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .profile-info h4 { margin: 5px 0; font-size: 18px; color: #222; font-weight: 700; }
    
    .profile-info p { margin: 0; font-size: 13px; color: #666; line-height: 1.4; }

    /* Personal Badges Container */
    .social-badges { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 10px; }
    .social-badges a img { height: 26px; border-radius: 4px; transition: transform 0.2s, opacity 0.2s; }
    .social-badges a img:hover { transform: translateY(-2px); opacity: 0.9; }
</style>
"""

footer_html = f"""
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
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank"><img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID"></a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=en&oi=ao" target="_blank"><img src="https://img.shields.io/badge/Scholar-Profile-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar"></a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank"><img src="https://img.shields.io/badge/ResearchGate-Profile-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate"></a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank"><img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes CV"></a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank"><img src="https://img.shields.io/badge/Web_of_Science-Profile-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science"></a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank"><img src="https://img.shields.io/badge/UNICAMP-Institutional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP"></a>
    </div>
</div>
"""

components.html(footer_css + footer_html, height=280, scrolling=False)
