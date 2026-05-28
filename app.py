"""                                                                                                                                        
                                      ++++++              ++++++                                    
                                      ++++++++          ++++++++                                    
                                      ++++++++          ++++++++                                    
                                      ++++++++++      ++++++++++                                    
                        --++          ++++++++++++++++++++++++++          ++++                      
                      --++++++        ++++++++++++++++++++++++++        ++++++++                    
                      ++++++++++::++++++++++++++++++++++++++++++++++--++++++++++                    
                        ++++++++++++++++++++++--      --++++++++++++++++++++++--                    
                        ++++++++++++++++--##################..++++++++++++++++                      
                          ++++++++++++########          ########++++++++++++                        
                          ++++++++++######                  ######::++++++++                        
                        --++++++++####        ##  ####    ..    ####++++++++++                      
              ++++::    ++++++++####      ####@@    ##    --  ##  ####++++++++    ::++++            
              ++++++++++++++++  ####      ##              ##mmMM  ####  ++++++++++++++++            
              ++++++++++++++++####                            ##    ####++++++++++++++++            
              ++++++++++++++::####      ##@@              ####      ####..++++++++++++++            
                  ++++++++++..##          ##                          ##  ++++++++++                
                      ++++++####          ++##        ####@@          ####++++++                    
                      ++++++####                  ############        ####++++++..                  
                      ++++++####          ####    ##############      ####++++++                    
                    ++++++++####        ####    ################      ##++++++++++                  
                ++++++++++++  ##        ##      ################    --##  ++++++++++++              
            ::++++++++++++++++####                ######    ##..    ####++++++++++++++++--          
              ++++++++++++++++####    --##        ############      ####++++++++++++++++            
              ++++++++++++++++::####    ##++          ####        ####--++++++++++++++++            
                        ++++++++  ####    ##                    ####  ++++++++                      
                          ++++++++MM####                      ####--++++++++                        
                          ++++++++++  ######              ######  ++++++++++                        
                          ++++++++++++  ######################  ++++++++++++                        
                        ++++++++++++++++++    ##########    ++++++++++++++++++                      
                      ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                    
                      ++++++++++    ++++++++++++++++++++++++++++++    ++++++++++                    
                        ++++++        ++++++++++++++++++++++++++        --++++                      
                                      ++++++++++++++++++++++++++                                    
                                      ++++++++++      ++++++++++                                    
                                      ++++++++          ++++++++                                    
                                      ++++++++          ++++++++                                    
                                          ++              ++--                                                                                                                                                                                                          


#############################################################################################
#                                                                                           #
#                                    GBMA - FEAGRI - UNICAMP                                #
#      -------------------------------------------------------------------------------      #
#                                                                                           #
#                         Interdisciplinary Research Group on Biotechnology                 #
#                           Applied to the Agriculture and the Environment                  #
#                                                                                           #
#                             School of Agricultural Engineering                            #
#                                    University of Campinas                                 #
#                                                                                           #
#############################################################################################

DEV: Prof. Dr. Gustavo Mockaitis
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
import io
import os
import hashlib
import socket
import base64
from datetime import datetime
import re
import random
import smtplib
import ssl
from email.mime.text import MIMEText
from copy import deepcopy
# from streamlit.web.server.websocket_headers import _get_websocket_headers ###Outdated
import uuid

# Import Core Logic from your separated file
from polyauxic_core import (
    boltzmann_term_eq31,
    gompertz_term_eq32,
    polyauxic_model,
    fit_model_auto,
    fit_model_auto_robust_pre,
    process_data,
    evaluate_polyauxic_model,
    first_order_term_phase1,
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
        headers = st.context.headers
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

def normalize_email(email):
    return str(email or "").strip().lower()

def validate_email_format(email):
    pattern = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
    return re.match(pattern, normalize_email(email)) is not None

def get_usage_registry_path():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "usage_registry.csv")

def append_usage_registry(event_type, profile, extra=None):
    """
    Stores onboarding/validation/upload data in a single local spreadsheet-like file.
    """
    profile = profile or {}
    extra = extra or {}
    registry_path = get_usage_registry_path()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "email": normalize_email(profile.get("email", "")),
        "first_name": str(profile.get("first_name", "")).strip(),
        "last_name": str(profile.get("last_name", "")).strip(),
        "institution": str(profile.get("institution", "")).strip(),
        "country": str(profile.get("country", "")).strip(),
        "experiment_description": str(profile.get("experiment_description", "")).strip(),
        "contact_opt_out": bool(profile.get("contact_opt_out", False)),
        "user_identifier": str(get_user_identifier()),
        "extra_json": str(extra)
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(registry_path):
        df_row.to_csv(registry_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(registry_path, mode="w", header=True, index=False)

def load_verified_profiles():
    registry_path = get_usage_registry_path()
    if not os.path.exists(registry_path):
        return {}
    try:
        df = pd.read_csv(registry_path)
    except Exception:
        return {}
    if df.empty or "event_type" not in df.columns or "email" not in df.columns:
        return {}
    validated = df[df["event_type"] == "otp_validated"].copy()
    if validated.empty:
        return {}
    validated["email"] = validated["email"].astype(str).str.lower().str.strip()
    validated = validated.sort_values("timestamp")
    latest = validated.groupby("email", as_index=False).tail(1)
    profiles = {}
    for _, row in latest.iterrows():
        email = str(row.get("email", "")).strip().lower()
        if not email:
            continue
        profiles[email] = {
            "email": email,
            "first_name": str(row.get("first_name", "")).strip(),
            "last_name": str(row.get("last_name", "")).strip(),
            "institution": str(row.get("institution", "")).strip(),
            "country": str(row.get("country", "")).strip(),
            "experiment_description": str(row.get("experiment_description", "")).strip(),
            "contact_opt_out": bool(row.get("contact_opt_out", False))
        }
    return profiles

def is_email_already_validated(email):
    validated_profiles = load_verified_profiles()
    return normalize_email(email) in validated_profiles

def send_otp_email(recipient_email, otp_code, lang):
    """
    Sends OTP code using SMTP environment variables.
    Required env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
    Optional env var:
      SMTP_USE_TLS=true|false (default true)
    """
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587").strip())
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    smtp_from = os.getenv("SMTP_FROM", "").strip() or smtp_user
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in ("1", "true", "yes", "on")

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, smtp_from]):
        return False, "SMTP not configured"

    subject_map = {
        "en": "Your Polyauxic Platform verification code",
        "pt": "Seu código de verificação da Plataforma Poliauxica",
        "fr": "Votre code de vérification de la plateforme polyauxique"
    }
    body_map = {
        "en": f"Your verification code is: {otp_code}\nThis code expires in 10 minutes.",
        "pt": f"Seu código de verificação é: {otp_code}\nEste código expira em 10 minutos.",
        "fr": f"Votre code de vérification est : {otp_code}\nCe code expire dans 10 minutes."
    }

    msg = MIMEText(body_map.get(lang, body_map["en"]), "plain", "utf-8")
    msg["Subject"] = subject_map.get(lang, subject_map["en"])
    msg["From"] = smtp_from
    msg["To"] = normalize_email(recipient_email)

    try:
        if smtp_use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, [msg["To"]], msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, [msg["To"]], msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)

def save_uploaded_data(df, user_profile=None):
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
        append_usage_registry(
            "data_upload",
            user_profile or {},
            {"saved_file": target_filename, "rows": int(len(df)), "columns": int(df.shape[1])}
        )
    except Exception as e:
        st.warning(f"Failed to save local backup file. Process will continue. Error: {e}")


def render_html_iframe(html_content, height=300, scrolling=False, force_iframe=False):
    """Render HTML content with st.html when possible, or iframe when required."""
    body = str(html_content)
    if isinstance(height, (int, float)) and height > 0:
        overflow_y = "auto" if scrolling else "hidden"
        body = f'<div style="height:{int(height)}px; overflow-y:{overflow_y}; overflow-x:hidden;">{body}</div>'

    # Some third-party badge scripts behave more reliably in iframe context.
    if not force_iframe and hasattr(st, "html"):
        try:
            st.html(body, width="stretch", unsafe_allow_javascript=True)
            return
        except TypeError:
            try:
                st.html(body, width="stretch")
                return
            except TypeError:
                st.html(body)
                return
        except Exception:
            pass

    encoded_html = base64.b64encode(body.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{encoded_html}", height=height)

# ==============================================================================
# EXCEL EXPORT UTILITY
# ==============================================================================

def generate_excel_report(best_results_global, replicates, param_labels, rate_label, lang):
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
            model_full_label = get_full_model_label(res, model_name, lang)
            use_first_order_phase1 = bool(res.get("use_first_order_phase1", False))
            
            def _fmt_pm_excel(v, se_v):
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                    if isinstance(se_v, (int, float, np.integer, np.floating)) and np.isfinite(se_v):
                        return f"{float(v):.6g} ± {float(se_v):.6g}"
                    return f"{float(v):.6g}"
                return str(v)

            # --- Global Parameters Sheet ---
            global_params = pd.DataFrame({
                "Parameter": [yi_name, yf_name, "Phases (n)", "Phase 1 Structure", "Full Model"],
                "Value ± SE": [
                    _fmt_pm_excel(theta[0], se[0]),
                    _fmt_pm_excel(theta[1], se[1]),
                    n,
                    "First-order" if res.get("use_first_order_phase1", False) else "Sigmoidal",
                    model_full_label
                ],
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
                is_first_order_phase1 = use_first_order_phase1 and i == 0
                lambda_display = "N/A" if is_first_order_phase1 else _fmt_pm_excel(lambda_[i], lambda_se[i])
                lambda_order = (
                    float("-inf")
                    if is_first_order_phase1
                    else (float(lambda_[i]) if np.isfinite(lambda_[i]) else np.inf)
                )
                phase_data.append({
                    "Phase": i + 1,
                    "Full Model": model_full_label,
                    "p ± SE": _fmt_pm_excel(p[i], se_p[i]),
                    f"{rate_label} ± SE": _fmt_pm_excel(r_max[i], r_max_se[i]),
                    "lambda ± SE": lambda_display,
                    "_lambda_order": lambda_order
                })
            
            phase_df = pd.DataFrame(phase_data)
            # Sort phases sequentially by lambda (lag time)
            phase_df = phase_df.sort_values(by="_lambda_order").drop(columns=["_lambda_order"]).reset_index(drop=True)
            phase_df.to_excel(writer, sheet_name=f"{model_name}_Phases", index=False)
            
            # --- Statistics & Metrics Sheet ---
            m = res['metrics']
            metrics_df = pd.DataFrame({
                "Metric": ["Full Model", "Correlation (r)", "R-squared", "Adjusted R-squared", "SSE", "AIC", "AICc", "BIC"],
                "Value": [model_full_label, m.get("r", np.nan), m['R2'], m['R2_adj'], m['SSE'], m['AIC'], m['AICc'], m['BIC']]
            })
            metrics_df.to_excel(writer, sheet_name=f"{model_name}_Metrics", index=False)

    return output.getvalue()


# ==============================================================================
# 0. CONFIGURATION & GLOBAL SETTINGS
# ==============================================================================

# Resolve fonts from what is actually installed in the runtime to avoid noisy findfont warnings.
AVAILABLE_MPL_FONT_NAMES = {font.name for font in fm.fontManager.ttflist}

def _first_available_font(candidates):
    for candidate in candidates:
        if candidate in AVAILABLE_MPL_FONT_NAMES:
            return candidate
    return None

DEFAULT_PLOT_FONT = _first_available_font(["DejaVu Serif", "DejaVu Sans", "Arial", "Calibri", "Verdana"]) or "DejaVu Sans"
DEFAULT_SANS_FONT = _first_available_font(["DejaVu Sans", "Arial", "Calibri", "Verdana"]) or DEFAULT_PLOT_FONT
DEFAULT_SERIF_FONT = _first_available_font(["DejaVu Serif", "Book Antiqua", "Bookman Old Style", DEFAULT_PLOT_FONT]) or DEFAULT_PLOT_FONT

plt.rcParams.update({
    'font.family': DEFAULT_SERIF_FONT,
    'font.sans-serif': [DEFAULT_SANS_FONT, 'DejaVu Sans', 'sans-serif'],
    'font.serif': [DEFAULT_SERIF_FONT, 'DejaVu Serif', 'serif'],
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
        * **Access:** Platform access requires mandatory onboarding form completion and email validation (OTP).
        * **Plot Settings:** You can customize plot font and axis titles in the sidebar.

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
        * **Acesso:** O uso da plataforma exige preenchimento do formulário inicial e validação do e-mail (OTP).
        * **Gráficos:** É possível definir a fonte e títulos dos eixos na barra lateral.

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
        * **Accès :** L’accès à la plateforme exige le formulaire initial et la validation de l’e-mail (OTP).
        * **Graphiques :** Vous pouvez personnaliser la police et les titres des axes dans la barre latérale.

        **Exemple de mise en page:**
        | A (Temps 1) | B (Rep 1) | C (Temps 2) | D (Rep 2) |
        | :--- | :--- | :--- | :--- |
        | 0.0 | 0.105 | 0.0 | 0.102 |
        | 1.0 | 0.200 | 1.0 | 0.198 |
        """
    },
    "sidebar_config": {"en": "Settings", "pt": "Configurações", "fr": "Paramètres"},
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
    "table_col_phase": {"en": "Phase", "pt": "Fase", "fr": "Phase"},
    "onboarding_title": {
        "en": "Welcome to the Polyauxic Modeling Platform",
        "pt": "Bem-vindo à Plataforma de Modelagem Poliauxica",
        "fr": "Bienvenue sur la plateforme de modélisation polyauxique"
    },
    "onboarding_subtitle": {
        "en": "Please select your language and complete the required form. Access is enabled only after email verification.",
        "pt": "Selecione seu idioma e preencha o formulário obrigatório. O acesso só é liberado após a validação do e-mail.",
        "fr": "Veuillez sélectionner votre langue et remplir le formulaire obligatoire. L’accès est activé uniquement après validation de l’e-mail."
    },
    "data_use_notice": {
        "en": "By using this platform, you acknowledge that uploaded/entered data may be collected for model training improvements, and contact data may be used for potential future collaborations.",
        "pt": "Ao usar esta plataforma, você reconhece que os dados enviados/inseridos podem ser coletados para melhorias de treinamento do modelo, e os dados de contato podem ser usados para potenciais colaborações futuras.",
        "fr": "En utilisant cette plateforme, vous reconnaissez que les données soumises/saisies peuvent être collectées pour améliorer l’entraînement du modèle, et que les données de contact peuvent être utilisées pour d’éventuelles collaborations futures."
    },
    "onboarding_card_instructions_title": {"en": "Instructions", "pt": "Instruções", "fr": "Instructions"},
    "onboarding_card_profile_title": {"en": "1) Profile", "pt": "1) Perfil", "fr": "1) Profil"},
    "onboarding_card_verify_title": {"en": "2) Access Verification", "pt": "2) Verificação de Acesso", "fr": "2) Vérification d’Accès"},
    "onboarding_card_instructions_body": {
        "en": "Complete all required profile fields and provide your email. If your email is already in the validated database, you can enter directly. Otherwise, send and verify the OTP code to unlock access.",
        "pt": "Preencha todos os campos obrigatórios do perfil e informe seu e-mail. Se o e-mail já estiver na base validada, você poderá entrar diretamente. Caso contrário, envie e valide o código OTP para liberar o acesso.",
        "fr": "Remplissez tous les champs obligatoires du profil et fournissez votre e-mail. Si votre e-mail est déjà dans la base validée, vous pouvez entrer directement. Sinon, envoyez et validez le code OTP pour débloquer l’accès."
    },
    "contact_opt_out": {
        "en": "I do not wish to be contacted for future collaborations.",
        "pt": "Não desejo ser contactado para colaborações futuras.",
        "fr": "Je ne souhaite pas être contacté pour de futures collaborations."
    },
    "form_first_name": {"en": "First Name", "pt": "Nome", "fr": "Prénom"},
    "form_last_name": {"en": "Last Name", "pt": "Sobrenome", "fr": "Nom"},
    "form_institution": {"en": "Institution", "pt": "Instituição", "fr": "Institution"},
    "form_country": {"en": "Country", "pt": "País", "fr": "Pays"},
    "form_description": {
        "en": "Brief experiment description",
        "pt": "Breve descrição do(s) experimento(s)",
        "fr": "Brève description de(s) expérience(s)"
    },
    "form_email": {"en": "Email", "pt": "E-mail", "fr": "E-mail"},
    "form_submit": {"en": "Continue", "pt": "Continuar", "fr": "Continuer"},
    "required_fields_error": {
        "en": "All fields are mandatory.",
        "pt": "Todos os campos são obrigatórios.",
        "fr": "Tous les champs sont obligatoires."
    },
    "email_invalid": {
        "en": "Please enter a valid email address.",
        "pt": "Informe um e-mail válido.",
        "fr": "Veuillez saisir une adresse e-mail valide."
    },
    "otp_send_button": {"en": "Send verification code", "pt": "Enviar código de verificação", "fr": "Envoyer le code de vérification"},
    "enter_button": {"en": "Enter", "pt": "Entrar", "fr": "Entrer"},
    "verify_enter_button": {"en": "Verify code and enter", "pt": "Verificar código e entrar", "fr": "Vérifier le code et entrer"},
    "otp_sent_success": {
        "en": "Verification code sent. Please check your inbox.",
        "pt": "Código de verificação enviado. Verifique sua caixa de entrada.",
        "fr": "Code de vérification envoyé. Veuillez vérifier votre boîte de réception."
    },
    "otp_send_fail": {
        "en": "Failed to send verification code: {0}",
        "pt": "Falha ao enviar código de verificação: {0}",
        "fr": "Échec de l’envoi du code de vérification : {0}"
    },
    "smtp_missing": {
        "en": "SMTP settings are not configured. Contact the administrator.",
        "pt": "Configurações SMTP não encontradas. Contate o administrador.",
        "fr": "Les paramètres SMTP ne sont pas configurés. Contactez l’administrateur."
    },
    "otp_code_label": {"en": "Verification code", "pt": "Código de verificação", "fr": "Code de vérification"},
    "otp_verify_button": {"en": "Validate email", "pt": "Validar e-mail", "fr": "Valider l’e-mail"},
    "otp_invalid": {
        "en": "Invalid or expired code.",
        "pt": "Código inválido ou expirado.",
        "fr": "Code invalide ou expiré."
    },
    "otp_validated": {
        "en": "Email successfully validated.",
        "pt": "E-mail validado com sucesso.",
        "fr": "E-mail validé avec succès."
    },
    "welcome_back_verified": {
        "en": "Email previously validated. Access granted.",
        "pt": "E-mail já validado anteriormente. Acesso liberado.",
        "fr": "E-mail déjà validé précédemment. Accès autorisé."
    },
    "returning_user_hint": {
        "en": "If this email has already used the platform, verification will not be required.",
        "pt": "Se este e-mail já utilizou a plataforma, a verificação não será necessária.",
        "fr": "Si cet e-mail a déjà utilisé la plateforme, la vérification ne sera pas nécessaire."
    },
    "access_blocked_until_validation": {
        "en": "Access remains blocked until email validation is completed.",
        "pt": "O acesso permanece bloqueado até concluir a validação do e-mail.",
        "fr": "L’accès reste bloqué jusqu’à la validation de l’e-mail."
    },
    "plot_font_label": {"en": "Plot Font", "pt": "Fonte dos Gráficos", "fr": "Police des Graphiques"},
    "seed_mode_label": {"en": "Random Seed Mode", "pt": "Modo da Seed Aleatória", "fr": "Mode de Seed Aléatoire"},
    "seed_fixed_42": {"en": "Fixed seed (42)", "pt": "Seed fixa (42)", "fr": "Seed fixe (42)"},
    "seed_random": {"en": "Random seed (new each run)", "pt": "Seed aleatória (nova a cada execução)", "fr": "Seed aléatoire (nouvelle à chaque exécution)"},
    "seed_used_msg": {"en": "Seed used in this run: {0}", "pt": "Seed usada nesta execução: {0}", "fr": "Seed utilisée dans cette exécution : {0}"},
    "axis_x_custom_label": {
        "en": "X-axis title (LaTeX accepted; blank = default)",
        "pt": "Título do eixo X (aceita LaTeX; vazio = padrão)",
        "fr": "Titre de l’axe X (LaTeX accepté; vide = défaut)"
    },
    "axis_y_custom_label": {
        "en": "Y-axis title (LaTeX accepted; blank = default)",
        "pt": "Título do eixo Y (aceita LaTeX; vazio = padrão)",
        "fr": "Titre de l’axe Y (LaTeX accepté; vide = défaut)"
    },
    "axis_math_hint": {
        "en": "Superscript examples: lambda^2, λ^2, \\lambda^2, or $\\lambda^2$.",
        "pt": "Exemplos de sobrescrito: lambda^2, λ^2, \\lambda^2 ou $\\lambda^2$.",
        "fr": "Exemples d’exposant : lambda^2, λ^2, \\lambda^2 ou $\\lambda^2$."
    },
    "default_y_label": {"en": "Response (y)", "pt": "Resposta (y)", "fr": "Réponse (y)"},
    "metric_corr": {"en": "Correlation (r)", "pt": "Correlação (r)", "fr": "Corrélation (r)"},
    "phase1_col": {"en": "Phase 1", "pt": "Fase 1", "fr": "Phase 1"},
    "phase1_first_order": {"en": "First-order", "pt": "Primeira ordem", "fr": "Premier ordre"},
    "phase1_sigmoidal": {"en": "Sigmoidal", "pt": "Sigmoidal", "fr": "Sigmoïdal"},
    "full_model_sigmoidal": {
        "en": "Sigmoidal ({0})",
        "pt": "Sigmoidal ({0})",
        "fr": "Sigmoïdal ({0})"
    },
    "full_model_first_order": {
        "en": "First-order + {0} (phase 1)",
        "pt": "1ª ordem + {0} (fase 1)",
        "fr": "Premier ordre + {0} (phase 1)"
    },
    "full_model_first_order_only": {
        "en": "First-order",
        "pt": "1ª ordem",
        "fr": "Premier ordre"
    },
    "full_model_col": {"en": "Full Model", "pt": "Modelo Completo", "fr": "Modèle Complet"},
    "fit_standard_header": {
        "en": "Standard Sigmoidal Fit",
        "pt": "Ajuste Sigmoidal Padrão",
        "fr": "Ajustement Sigmoïdal Standard"
    },
    "fit_first_order_header": {
        "en": "Additional Phase-1 First-Order Fit",
        "pt": "Ajuste Adicional com 1ª Fase de Primeira Ordem",
        "fr": "Ajustement Supplémentaire avec 1ʳᵉ Phase de Premier Ordre"
    },
    "fit_reused_n1": {
        "en": "Reused from {0} single-phase first-order fit (no recomputation).",
        "pt": "Reutilizado do ajuste de fase única em 1ª ordem de {0} (sem novo cálculo).",
        "fr": "Réutilisé depuis l’ajustement mono-phase de premier ordre de {0} (sans recalcul)."
    },
    "fit_overview_header": {
        "en": "Fit Overview",
        "pt": "Visão Geral do Ajuste",
        "fr": "Aperçu de l’Ajustement"
    },
    "details_expander": {
        "en": "Show detailed parameters and diagnostics",
        "pt": "Mostrar parâmetros detalhados e diagnósticos",
        "fr": "Afficher les paramètres détaillés et diagnostics"
    },
    "selection_logic_title": {
        "en": "Automatic Selection Logic",
        "pt": "Lógica Automática de Seleção",
        "fr": "Logique de Sélection Automatique"
    },
    "selection_logic_body": {
        "en": "Criterion in use: **{0}**. For each phase count, the best candidate between available structures is selected. Then the final phase count follows the first local minimum rule of the selected criterion.",
        "pt": "Critério em uso: **{0}**. Para cada número de fases, o melhor candidato entre as estruturas disponíveis é selecionado. Em seguida, o número final de fases segue a regra do primeiro mínimo local do critério selecionado.",
        "fr": "Critère utilisé : **{0}**. Pour chaque nombre de phases, le meilleur candidat parmi les structures disponibles est sélectionné. Ensuite, le nombre final de phases suit la règle du premier minimum local du critère sélectionné."
    },
    "legend_phase_best": {
        "en": "Green rows: best candidate within each phase count.",
        "pt": "Linhas verdes: melhor candidato dentro de cada número de fases.",
        "fr": "Lignes vertes : meilleur candidat pour chaque nombre de phases."
    },
    "legend_final_best": {
        "en": "Blue row: final selected model.",
        "pt": "Linha azul: modelo final selecionado.",
        "fr": "Ligne bleue : modèle final sélectionné."
    },
    "metrics_marker_sigmoidal": {
        "en": "Sigmoidal phase-1 structure",
        "pt": "Estrutura sigmoidal na fase 1",
        "fr": "Structure sigmoïdale en phase 1"
    },
    "metrics_marker_first_order": {
        "en": "First-order phase-1 structure",
        "pt": "Estrutura de primeira ordem na fase 1",
        "fr": "Structure de premier ordre en phase 1"
    },
    "main_section_adjustments_results": {
        "en": "Adjustments and Results",
        "pt": "Ajustes e Resultados",
        "fr": "Ajustements et Résultats"
    },
    "graphic_options_header": {"en": "Graphic Options", "pt": "Opções de Gráficos", "fr": "Options Graphiques"},
    "main_language_label": {"en": "Language", "pt": "Idioma", "fr": "Langue"}
}

COUNTRY_OPTIONS = [
    "Argentina", "Australia", "Austria", "Belgium", "Bolivia", "Brazil", "Bulgaria", "Canada",
    "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cuba", "Czech Republic", "Denmark",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia", "Finland", "France",
    "Germany", "Greece", "Guatemala", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Ireland", "Israel", "Italy", "Japan", "Kenya", "Latvia", "Lithuania", "Luxembourg",
    "Malaysia", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nicaragua", "Nigeria",
    "Norway", "Pakistan", "Panama", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Romania", "Russia", "Saudi Arabia", "Serbia", "Singapore", "Slovakia", "Slovenia",
    "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Venezuela",
    "Vietnam", "Other"
]

_BASE_PLOT_FONT_OPTIONS = [
    "DejaVu Serif",
    "DejaVu Sans",
    "Arial",
    "Calibri",
    "Book Antiqua",
    "Bookman Old Style",
    "Verdana",
    "Times New Roman"
]
# Keep all requested fonts visible in the UI. Runtime fallback is handled in resolve_plot_font.
PLOT_FONT_OPTIONS = list(_BASE_PLOT_FONT_OPTIONS)
if DEFAULT_PLOT_FONT not in PLOT_FONT_OPTIONS:
    PLOT_FONT_OPTIONS.insert(0, DEFAULT_PLOT_FONT)
if DEFAULT_PLOT_FONT in PLOT_FONT_OPTIONS:
    PLOT_FONT_OPTIONS = [DEFAULT_PLOT_FONT] + [font for font in PLOT_FONT_OPTIONS if font != DEFAULT_PLOT_FONT]

def resolve_plot_font(font_name):
    candidate = str(font_name or "").strip()
    if candidate in AVAILABLE_MPL_FONT_NAMES:
        return candidate
    semantic_fallbacks = {
        "Times New Roman": ["DejaVu Serif", "Book Antiqua", "Bookman Old Style", "DejaVu Sans"],
        "Book Antiqua": ["DejaVu Serif", "Times New Roman", "Bookman Old Style", "DejaVu Sans"],
        "Bookman Old Style": ["DejaVu Serif", "Book Antiqua", "Times New Roman", "DejaVu Sans"],
        "Arial": ["DejaVu Sans", "Calibri", "Verdana", "DejaVu Serif"],
        "Calibri": ["DejaVu Sans", "Arial", "Verdana", "DejaVu Serif"],
        "Verdana": ["DejaVu Sans", "Arial", "Calibri", "DejaVu Serif"],
        "DejaVu Serif": ["DejaVu Sans"],
        "DejaVu Sans": ["DejaVu Serif"],
    }
    for fallback in semantic_fallbacks.get(candidate, []):
        if fallback in AVAILABLE_MPL_FONT_NAMES:
            return fallback
    return DEFAULT_PLOT_FONT

def apply_plot_font(font_name):
    selected = resolve_plot_font(font_name if font_name in PLOT_FONT_OPTIONS else DEFAULT_PLOT_FONT)
    plt.rcParams.update({
        "font.family": selected,
        "font.sans-serif": [selected, DEFAULT_SANS_FONT, "DejaVu Sans", "sans-serif"],
        "font.serif": [selected, DEFAULT_SERIF_FONT, "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 12,
        "mathtext.fontset": "stix"
    })

def get_current_plot_font():
    selected = st.session_state.get("plot_font_selector", DEFAULT_PLOT_FONT)
    return resolve_plot_font(selected)

def style_axes_fonts(ax, font_name=None):
    font_name = resolve_plot_font(font_name or get_current_plot_font())
    if ax.get_title():
        ax.set_title(ax.get_title(), fontname=font_name)
    ax.set_xlabel(ax.get_xlabel(), fontname=font_name)
    ax.set_ylabel(ax.get_ylabel(), fontname=font_name)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(font_name)
    leg = ax.get_legend()
    if leg:
        for txt in leg.get_texts():
            txt.set_fontname(font_name)

def normalize_axis_math_label(label_text):
    """
    Accepts user-friendly axis text and converts common superscript inputs
    to Matplotlib mathtext, e.g. lambda^2, lambdaˆ2, λ^2, <sup>2</sup>.
    """
    text = str(label_text or "").strip()
    if not text:
        return text

    # Common copy/paste variants seen in UI input.
    text = text.replace("ˆ", "^")
    text = re.sub(r"(?is)<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"^\1", text)

    def _to_unicode_superscript(exp_txt):
        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "+": "⁺",
            "-": "⁻",
        }
        converted = "".join(superscript_map.get(ch, ch) for ch in exp_txt)
        return converted

    def _convert_compact_unit_exponents(raw_text):
        # Converts unit-style tokens such as m2, m-2, cm3, h-1 to Unicode superscript.
        # Restrict base length to avoid changing ordinary words like "phase2".
        unit_pattern = re.compile(r"\b([A-Za-zµμΩωα-ωΑ-Ω]{1,4})([-+]?\d+)\b")

        def _repl(match):
            base = match.group(1)
            exponent = match.group(2)
            return f"{base}{_to_unicode_superscript(exponent)}"

        return unit_pattern.sub(_repl, raw_text)

    text = _convert_compact_unit_exponents(text)

    # If user already provided mathtext, keep it unchanged.
    if "$" in text:
        return text

    has_math_hint = ("^" in text) or ("_" in text) or any(ch in text for ch in ("λ", "μ", "σ", "θ", "φ", "ω"))
    has_greek_word = re.search(
        r"\b(lambda|alpha|beta|gamma|delta|epsilon|mu|sigma|theta|phi|omega)\b",
        text,
        flags=re.IGNORECASE
    ) is not None
    if not (has_math_hint or has_greek_word):
        return text

    # Replace common Greek names/symbols with mathtext commands.
    greek_symbols = {
        "λ": r"\lambda",
        "μ": r"\mu",
        "σ": r"\sigma",
        "θ": r"\theta",
        "φ": r"\phi",
        "ω": r"\omega",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
    }
    for symbol, latex_cmd in greek_symbols.items():
        text = text.replace(symbol, latex_cmd)

    greek_words = {
        "lambda": r"\lambda",
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "mu": r"\mu",
        "sigma": r"\sigma",
        "theta": r"\theta",
        "phi": r"\phi",
        "omega": r"\omega",
    }
    for word, latex_cmd in greek_words.items():
        text = re.sub(
            rf"\b{word}\b",
            lambda _m, rep=latex_cmd: rep,
            text,
            flags=re.IGNORECASE
        )

    # Convert compact exponent after Greek names too (e.g., lambda2 -> lambda^2).
    text = re.sub(
        r"(\\(?:lambda|alpha|beta|gamma|delta|epsilon|mu|sigma|theta|phi|omega))([-+]?\d+)\b",
        r"\1^\2",
        text
    )

    # Normalize simple exponents like ^2, ^-1, ^10 to brace form.
    text = re.sub(r"\^(-?\d+)", r"^{\1}", text)

    return f"${text}$"

def build_all_data_df(replicates):
    all_data = []
    for rep in replicates:
        for t, y in zip(rep["t"], rep["y"]):
            all_data.append({"t": float(t), "y": float(y)})
    if not all_data:
        return pd.DataFrame(columns=["t", "y"])
    return pd.DataFrame(all_data).sort_values("t").reset_index(drop=True)

def get_best_candidate_by_phase(results, ic_name):
    """
    From all candidates (including alternate phase-1 structures), keep the best per phase count.
    """
    if not results:
        return []
    grouped = {}
    for r in results:
        phase_n = int(r["n_phases"])
        grouped.setdefault(phase_n, []).append(r)

    best_by_phase = []
    for phase_n in sorted(grouped.keys()):
        phase_candidates = grouped[phase_n]
        best_phase = min(
            phase_candidates,
            key=lambda x: float(x["metrics"].get(ic_name, np.inf))
        )
        best_by_phase.append(best_phase)
    return best_by_phase


def get_full_model_label(res, model_name, lang):
    if res.get("use_first_order_phase1", False):
        if int(res.get("n_phases", 0)) == 1:
            return TEXTS["full_model_first_order_only"][lang]
        return TEXTS["full_model_first_order"][lang].format(model_name)
    return TEXTS["full_model_sigmoidal"][lang].format(model_name)

def format_subscript_label(label):
    """
    Converts underscore notation into subscript-like presentation where possible.
    Example: r_max -> rₘₐₓ
    """
    sub_map = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ"
    }
    txt = str(label)
    if "_" not in txt:
        return txt
    base, sub = txt.split("_", 1)
    sub_fmt = "".join(sub_map.get(ch, ch) for ch in sub.lower())
    return f"{base}{sub_fmt}"

def should_test_first_order_variant(res, tol=1e-6):
    if not res:
        return False
    n = int(res["n_phases"])
    lambda_ = res["theta"][2 + 2 * n : 2 + 3 * n]
    if len(lambda_) == 0:
        return False
    return abs(float(lambda_[0])) <= tol

def run_fit_with_outlier_strategy(
    t_flat,
    y_flat,
    model_func,
    n,
    outlier_method_key,
    force_yi=False,
    force_yf=False,
    rout_q=1.0,
    use_first_order_phase1=False,
    random_seed=42
):
    """
    Runs one fitting pass with a selected outlier strategy and returns a result on full data indexing.
    """
    n_params = 2 + 3 * n
    full_mask = np.zeros(len(y_flat), dtype=bool)
    res = None

    if outlier_method_key == "none":
        res = fit_model_auto(
            t_flat,
            y_flat,
            model_func,
            n,
            force_yi=force_yi,
            force_yf=force_yf,
            use_first_order_phase1=use_first_order_phase1,
            random_seed=random_seed
        )

    elif outlier_method_key == "simple":
        res_pre = fit_model_auto(
            t_flat,
            y_flat,
            model_func,
            n,
            force_yi=force_yi,
            force_yf=force_yf,
            use_first_order_phase1=use_first_order_phase1,
            random_seed=random_seed
        )
        if res_pre:
            y_pred_pre = evaluate_polyauxic_model(
                t_flat,
                res_pre["theta"],
                model_func,
                n,
                use_first_order_phase1=use_first_order_phase1
            )
            full_mask = detect_outliers(y_flat, y_pred_pre)
            if np.any(full_mask) and (len(y_flat[~full_mask]) > n_params + 5):
                res = fit_model_auto(
                    t_flat[~full_mask],
                    y_flat[~full_mask],
                    model_func,
                    n,
                    force_yi=force_yi,
                    force_yf=force_yf,
                    use_first_order_phase1=use_first_order_phase1,
                    random_seed=random_seed
                )
            else:
                res = res_pre
                full_mask = np.zeros(len(y_flat), dtype=bool)

    elif outlier_method_key == "rout":
        res_robust = fit_model_auto_robust_pre(
            t_flat,
            y_flat,
            model_func,
            n,
            force_yi=force_yi,
            force_yf=force_yf,
            use_first_order_phase1=use_first_order_phase1,
            random_seed=random_seed
        )
        if res_robust:
            y_pred_pre = res_robust["y_pred"]
            full_mask = detect_outliers_rout_rigorous(
                y_flat,
                y_pred_pre,
                n_params=n_params,
                Q=rout_q
            )
            if np.any(full_mask) and (len(y_flat[~full_mask]) > n_params + 5):
                res = fit_model_auto(
                    t_flat[~full_mask],
                    y_flat[~full_mask],
                    model_func,
                    n,
                    force_yi=force_yi,
                    force_yf=force_yf,
                    use_first_order_phase1=use_first_order_phase1,
                    random_seed=random_seed
                )
            else:
                res = fit_model_auto(
                    t_flat,
                    y_flat,
                    model_func,
                    n,
                    force_yi=force_yi,
                    force_yf=force_yf,
                    use_first_order_phase1=use_first_order_phase1,
                    random_seed=random_seed
                )
                full_mask = np.zeros(len(y_flat), dtype=bool)

    if res:
        res["outliers"] = full_mask
        res["t_clean"] = t_flat[~full_mask] if np.any(full_mask) else t_flat
        res["y_clean"] = y_flat[~full_mask] if np.any(full_mask) else y_flat
        res["y_pred_full"] = evaluate_polyauxic_model(
            t_flat,
            res["theta"],
            model_func,
            n,
            use_first_order_phase1=use_first_order_phase1
        )
        res["use_first_order_phase1"] = bool(use_first_order_phase1)
    return res

def render_developer_footer_card():
    profile_pic_url = "https://github.com/gusmock.png"
    footer_css = """
    <style>
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
        .profile-section {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            max-width: 900px;
        }
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
    render_html_iframe(footer_css + footer_html, height=280, scrolling=False)

def render_access_gate():
    """
    Mandatory onboarding + email verification gate.
    Returns selected language code once access is granted.
    """
    if "gate_lang_selector" not in st.session_state:
        st.session_state.gate_lang_selector = list(LANGUAGES.keys())[0]
    if "access_granted" not in st.session_state:
        st.session_state.access_granted = False

    if st.session_state.access_granted:
        lang_key = st.session_state.get("gate_lang_selector", list(LANGUAGES.keys())[0])
        lang = LANGUAGES.get(lang_key, "en")
        st.session_state["lang"] = lang
        return lang

    title_col, lang_col = st.columns([6, 1])
    with title_col:
        current_gate_lang = LANGUAGES.get(st.session_state.get("gate_lang_selector", list(LANGUAGES.keys())[0]), "en")
        st.title(TEXTS["onboarding_title"][current_gate_lang])
    with lang_col:
        lang_key = st.selectbox(
            "Language / Idioma / Langue",
            list(LANGUAGES.keys()),
            key="gate_lang_selector",
            label_visibility="collapsed"
        )
    lang = LANGUAGES[lang_key]
    st.session_state["lang"] = lang

    st.info(TEXTS["onboarding_subtitle"][lang])
    st.caption(TEXTS["data_use_notice"][lang])
    pending_profile = st.session_state.get("pending_profile", {})
    for k, default in [
        ("gate_first_name", pending_profile.get("first_name", "")),
        ("gate_last_name", pending_profile.get("last_name", "")),
        ("gate_institution", pending_profile.get("institution", "")),
        ("gate_country", pending_profile.get("country", "Other")),
        ("gate_experiment_description", pending_profile.get("experiment_description", "")),
        ("gate_email", pending_profile.get("email", "")),
        ("gate_contact_opt_out", bool(pending_profile.get("contact_opt_out", False))),
    ]:
        if k not in st.session_state:
            st.session_state[k] = default

    col_info, col_profile, col_verify = st.columns([1, 1, 1], gap="large")
    with col_info:
        with st.container(border=True):
            st.markdown(f"#### {TEXTS['onboarding_card_instructions_title'][lang]}")
            st.markdown(TEXTS["onboarding_card_instructions_body"][lang])
            st.caption(TEXTS["returning_user_hint"][lang])

    with col_profile:
        with st.container(border=True):
            st.markdown(f"#### {TEXTS['onboarding_card_profile_title'][lang]}")
            st.text_input(TEXTS["form_first_name"][lang], key="gate_first_name")
            st.text_input(TEXTS["form_last_name"][lang], key="gate_last_name")
            st.text_input(TEXTS["form_institution"][lang], key="gate_institution")
            country_value = st.session_state.get("gate_country", "Other")
            if country_value not in COUNTRY_OPTIONS:
                st.session_state["gate_country"] = "Other"
            st.selectbox(TEXTS["form_country"][lang], COUNTRY_OPTIONS, key="gate_country")
            st.text_area(TEXTS["form_description"][lang], key="gate_experiment_description")
            st.checkbox(TEXTS["contact_opt_out"][lang], key="gate_contact_opt_out")

    with col_verify:
        with st.container(border=True):
            st.markdown(f"#### {TEXTS['onboarding_card_verify_title'][lang]}")
            st.text_input(TEXTS["form_email"][lang], key="gate_email")
            email_norm = normalize_email(st.session_state.get("gate_email", ""))
            pending_email = normalize_email(st.session_state.get("otp_requested_email", ""))
            if pending_email and pending_email != email_norm:
                for key in ("pending_otp_hash", "pending_otp_expires", "pending_profile", "otp_requested_email", "gate_otp_input"):
                    if key in st.session_state:
                        del st.session_state[key]
            known_email = validate_email_format(email_norm) and is_email_already_validated(email_norm)
            awaiting_code = (
                ("pending_otp_hash" in st.session_state)
                and normalize_email(st.session_state.get("otp_requested_email", "")) == email_norm
            )

            known_profile = load_verified_profiles().get(email_norm, None) if known_email else None
            if known_profile:
                if not st.session_state.get("gate_first_name", "").strip():
                    st.session_state["gate_first_name"] = known_profile.get("first_name", "")
                if not st.session_state.get("gate_last_name", "").strip():
                    st.session_state["gate_last_name"] = known_profile.get("last_name", "")
                if not st.session_state.get("gate_institution", "").strip():
                    st.session_state["gate_institution"] = known_profile.get("institution", "")
                if not st.session_state.get("gate_experiment_description", "").strip():
                    st.session_state["gate_experiment_description"] = known_profile.get("experiment_description", "")
                if st.session_state.get("gate_country", "Other") == "Other" and known_profile.get("country") in COUNTRY_OPTIONS:
                    st.session_state["gate_country"] = known_profile.get("country")

            def _build_profile_from_gate():
                return {
                    "first_name": str(st.session_state.get("gate_first_name", "")).strip(),
                    "last_name": str(st.session_state.get("gate_last_name", "")).strip(),
                    "institution": str(st.session_state.get("gate_institution", "")).strip(),
                    "country": str(st.session_state.get("gate_country", "")).strip(),
                    "experiment_description": str(st.session_state.get("gate_experiment_description", "")).strip(),
                    "email": normalize_email(st.session_state.get("gate_email", "")),
                    "contact_opt_out": bool(st.session_state.get("gate_contact_opt_out", False))
                }

            def _validate_required_profile(profile):
                required = [
                    profile.get("first_name", ""),
                    profile.get("last_name", ""),
                    profile.get("institution", ""),
                    profile.get("country", ""),
                    profile.get("experiment_description", ""),
                    profile.get("email", "")
                ]
                if not all(bool(str(x).strip()) for x in required):
                    return False
                return True

            if known_email:
                if st.button(TEXTS["enter_button"][lang], width="stretch", key="gate_enter_btn"):
                    profile = _build_profile_from_gate()
                    if not _validate_required_profile(profile):
                        st.error(TEXTS["required_fields_error"][lang])
                        st.stop()
                    if not validate_email_format(profile["email"]):
                        st.error(TEXTS["email_invalid"][lang])
                        st.stop()
                    st.session_state["user_profile"] = profile
                    st.session_state["access_granted"] = True
                    append_usage_registry("validated_access", profile, {"source": "previously_validated"})
                    st.success(TEXTS["welcome_back_verified"][lang])
                    st.rerun()
            else:
                if not awaiting_code:
                    if st.button(TEXTS["otp_send_button"][lang], width="stretch", key="gate_send_otp_btn"):
                        profile = _build_profile_from_gate()
                        if not _validate_required_profile(profile):
                            st.error(TEXTS["required_fields_error"][lang])
                            st.stop()
                        if not validate_email_format(profile["email"]):
                            st.error(TEXTS["email_invalid"][lang])
                            st.stop()

                        otp_code = f"{random.randint(0, 999999):06d}"
                        ok, err = send_otp_email(profile["email"], otp_code, lang)
                        if not ok:
                            err_msg = TEXTS["smtp_missing"][lang] if err == "SMTP not configured" else err
                            st.error(TEXTS["otp_send_fail"][lang].format(err_msg))
                            st.stop()

                        st.session_state["pending_profile"] = profile
                        st.session_state["pending_otp_hash"] = hashlib.sha256(otp_code.encode("utf-8")).hexdigest()
                        st.session_state["pending_otp_expires"] = datetime.now().timestamp() + 600
                        st.session_state["otp_requested_email"] = profile["email"]
                        append_usage_registry("otp_sent", profile, {"valid_for_sec": 600})
                        st.success(TEXTS["otp_sent_success"][lang])
                        st.rerun()
                else:
                    st.text_input(TEXTS["otp_code_label"][lang], key="gate_otp_input")
                    if st.button(TEXTS["verify_enter_button"][lang], width="stretch", key="gate_verify_enter_btn"):
                        otp_input = str(st.session_state.get("gate_otp_input", "")).strip()
                        otp_hash = hashlib.sha256(otp_input.encode("utf-8")).hexdigest()
                        not_expired = datetime.now().timestamp() <= float(st.session_state.get("pending_otp_expires", 0))
                        if not_expired and otp_hash == st.session_state.get("pending_otp_hash"):
                            profile = _build_profile_from_gate()
                            st.session_state["user_profile"] = profile
                            st.session_state["access_granted"] = True
                            append_usage_registry("otp_validated", profile, {"status": "ok"})
                            for key in ("pending_otp_hash", "pending_otp_expires", "pending_profile", "otp_requested_email", "gate_otp_input"):
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success(TEXTS["otp_validated"][lang])
                            st.rerun()
                        else:
                            profile = _build_profile_from_gate()
                            append_usage_registry("otp_validation_failed", profile, {"status": "invalid_or_expired"})
                            st.error(TEXTS["otp_invalid"][lang])

    st.warning(TEXTS["access_blocked_until_validation"][lang])
    st.markdown("---")
    render_developer_footer_card()
    st.stop()

# ==============================================================================
# 4. VISUALIZATION & APP STRUCTURE
# ==============================================================================

def plot_raw_data(replicates, lang, x_label, y_label):
    """Plots raw data before analysis."""
    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    for rep in replicates:
        ax.scatter(rep['t'], rep['y'], facecolors='white', edgecolors='black', alpha=0.8, s=20)
    ax.set_title("Experimental Data", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=':', alpha=0.3)
    style_axes_fonts(ax)
    st.pyplot(fig)

def plot_final_summary(replicates, best_results, lang, x_label, y_label):
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
        y_smooth = evaluate_polyauxic_model(
            t_smooth,
            res['theta'],
            func,
            res['n_phases'],
            use_first_order_phase1=bool(res.get("use_first_order_phase1", False))
        )
        
        if res['metrics']['AICc'] < best_aic_val:
            best_aic_val = res['metrics']['AICc']
            best_overall_res = res
            
        label = f"{model_name}: {res['n_phases']} phases (AICc: {res['metrics']['AICc']:.1f})"
        ax.plot(t_smooth, y_smooth, linewidth=2, color=colors.get(model_name, 'black'), label=label)

    if best_overall_res is not None:
        outlier_count = np.sum(best_overall_res['outliers'])
        ax.set_title(f"Best Fit Summary (Outliers detected by best model: {outlier_count})", fontsize=12)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    style_axes_fonts(ax)
    st.pyplot(fig)


def plot_metrics_summary(results_list, model_name, lang):
    """Generates a summary chart of metrics vs phases."""
    phases = [r['n_phases'] for r in results_list]
    aic = [r['metrics']['AIC'] for r in results_list]
    aicc = [r['metrics']['AICc'] for r in results_list]
    bic = [r['metrics']['BIC'] for r in results_list]
    r2_adj = [r['metrics']['R2_adj'] for r in results_list]
    phase1_structure = [bool(r.get("use_first_order_phase1", False)) for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(phases, aic, 'o--', label='AIC')
    ax1.plot(phases, aicc, 's-', label='AICc')
    ax1.plot(phases, bic, '^:', label='BIC')
    for x, y, is_first_order in zip(phases, aicc, phase1_structure):
        marker = "D" if is_first_order else "o"
        face = "tab:blue" if is_first_order else "white"
        ax1.scatter([x], [y], marker=marker, s=70, c=face, edgecolors='black', zorder=5)
    ax1.set_xlabel('Number of Phases')
    ax1.set_ylabel('Value')
    ax1.set_title('Information Criteria')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(phases, r2_adj, 'o-', label='Adjusted R²')
    for x, y, is_first_order in zip(phases, r2_adj, phase1_structure):
        marker = "D" if is_first_order else "o"
        face = "tab:blue" if is_first_order else "white"
        ax2.scatter([x], [y], marker=marker, s=70, c=face, edgecolors='black', zorder=5)
    ax2.set_xlabel('Number of Phases')
    ax2.set_ylabel('Adjusted R²')
    ax2.set_title('Fit Quality')
    marker_legend = [
        Line2D([], [], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', label=TEXTS["metrics_marker_sigmoidal"][lang]),
        Line2D([], [], marker='D', linestyle='None', markerfacecolor='tab:blue', markeredgecolor='black', label=TEXTS["metrics_marker_first_order"][lang]),
    ]
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + marker_legend)
    ax2.grid(True, alpha=0.3)
    style_axes_fonts(ax1)
    style_axes_fonts(ax2)
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

def display_single_fit(res, replicates, model_name, model_func, color_main, x_label, y_label, param_labels, rate_label, lang):
    """Displays detailed results for a single fit."""
    n = res['n_phases']
    theta = res['theta']
    se = res['se']
    se_p = res['se_p']
    yi_name, yf_name = param_labels
    full_model_label = get_full_model_label(res, model_name, lang)
    phase1_label = TEXTS["phase1_first_order"][lang] if res.get("use_first_order_phase1", False) else TEXTS["phase1_sigmoidal"][lang]
    raw_data_w_outliers = build_all_data_df(replicates)
    outliers_mask = np.asarray(res.get("outliers", np.zeros(len(raw_data_w_outliers), dtype=bool)), dtype=bool)
    if len(outliers_mask) != len(raw_data_w_outliers):
        outliers_mask = np.zeros(len(raw_data_w_outliers), dtype=bool)
    raw_data_w_outliers["is_outlier"] = outliers_mask
    raw_data_w_outliers["t_round"] = raw_data_w_outliers["t"].round(4)
    stats_df = raw_data_w_outliers[~raw_data_w_outliers["is_outlier"]].groupby("t_round")["y"].agg(["mean", "std"]).reset_index()
    y_i, y_f = theta[0], theta[1]
    y_i_se, y_f_se = se[0], se[1]

    z = theta[2 : 2 + n]
    r_max = theta[2 + n : 2 + 2 * n]
    r_max_se = se[2 + n : 2 + 2 * n]
    lambda_ = theta[2 + 2 * n : 2 + 3 * n]
    lambda_se = se[2 + 2 * n : 2 + 3 * n]
    p = np.exp(z - np.max(z))
    p /= np.sum(p)

    use_first_order_phase1 = bool(res.get("use_first_order_phase1", False))
    phases = []
    for i in range(n):
        phases.append({
            "phase_id": i,
            "p": p[i],
            "SE p": se_p[i],
            "r_max": r_max[i],
            "r_max_se": r_max_se[i],
            "lambda": lambda_[i],
            "lambda_se": lambda_se[i]
        })
    # Keep phase-1 first when it is first-order (lambda_1 is not part of that equation).
    phases.sort(
        key=lambda x: (
            0 if (use_first_order_phase1 and x.get("phase_id", -1) == 0) else 1,
            x["lambda"]
        )
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

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

    if len(replicates) > 1 and not stats_df.empty:
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
    y_smooth = evaluate_polyauxic_model(
        t_smooth,
        theta,
        model_func,
        n,
        use_first_order_phase1=bool(res.get("use_first_order_phase1", False))
    )
    ax.plot(t_smooth, y_smooth, color=color_main, linewidth=2.5, label=TEXTS['legend_global'][lang])

    colors = plt.cm.viridis(np.linspace(0, 0.9, n))
    for i, ph in enumerate(phases):
        if use_first_order_phase1 and ph.get("phase_id", -1) == 0:
            y_ind = first_order_term_phase1(t_smooth, ph['p'], y_f, ph['r_max'])
        else:
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    style_axes_fonts(ax)
    download_key = f"dl_btn_{model_name.lower()}_{model_func.__name__}_{n}_{'firstorder' if res.get('use_first_order_phase1', False) else 'standard'}"

    left_col, right_col = st.columns([2.4, 1.2], gap="large")
    with left_col:
        st.download_button(
            label=TEXTS['download_plot'][lang],
            data=buf.getvalue(),
            file_name=f"plot_{n}_phases.svg",
            mime="image/svg+xml",
            key=download_key
        )
        st.pyplot(fig)

    m = res['metrics']
    def _fmt_mixed(v):
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            return f"{float(v):.4f}"
        if isinstance(v, (int, float, np.integer, np.floating)) and not np.isfinite(v):
            return "N/A"
        return v
    with right_col:
        st.markdown(f"**{TEXTS['fit_overview_header'][lang]}**")
        df_overview = pd.DataFrame(
            {
                TEXTS['table_col_metric'][lang]: [
                    TEXTS["full_model_col"][lang],
                    TEXTS["phase1_col"][lang],
                    "F",
                    TEXTS['metric_corr'][lang],
                    "R²",
                    "R² Adj",
                    "SSE",
                    "AIC",
                    "AICc",
                    "BIC"
                ],
                TEXTS['table_col_value'][lang]: [
                    full_model_label,
                    phase1_label,
                    n,
                    m.get('r', np.nan),
                    m['R2'],
                    m['R2_adj'],
                    m['SSE'],
                    m['AIC'],
                    m['AICc'],
                    m['BIC']
                ]
            }
        )
        df_overview[TEXTS['table_col_value'][lang]] = df_overview[TEXTS['table_col_value'][lang]].map(_fmt_mixed)
        st.dataframe(df_overview, hide_index=True, width="stretch")

    with st.expander(TEXTS["details_expander"][lang], expanded=False):
        def _fmt_pm(v, se_v):
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                if isinstance(se_v, (int, float, np.integer, np.floating)) and np.isfinite(se_v):
                    return f"{float(v):.4f} ± {float(se_v):.4f}"
                return f"{float(v):.4f}"
            return str(v)

        df_glob = pd.DataFrame(
            {
                TEXTS['table_col_param'][lang]: [yi_name, yf_name],
                f"{TEXTS['table_col_value'][lang]} ± {TEXTS['table_col_se'][lang]}": [
                    _fmt_pm(y_i, y_i_se),
                    _fmt_pm(y_f, y_f_se)
                ]
            }
        )
        st.dataframe(df_glob, hide_index=True, width="stretch")

        rows = []
        for i, ph in enumerate(phases):
            lambda_display = (
                "N/A"
                if (use_first_order_phase1 and ph.get("phase_id", -1) == 0)
                else _fmt_pm(ph['lambda'], ph['lambda_se'])
            )
            rows.append({
                TEXTS['table_col_phase'][lang]: i + 1,
                "p ± SE": _fmt_pm(ph['p'], ph['SE p']),
                f"{rate_label} ± {TEXTS['table_col_se'][lang]}": _fmt_pm(ph['r_max'], ph['r_max_se']),
                "λ ± SE": lambda_display
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch"
        )

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
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    if "run_seed" not in st.session_state:
        st.session_state.run_seed = 42

    lang = render_access_gate()
    current_main_lang_key = st.session_state.get("main_lang_selector", None)
    if current_main_lang_key in LANGUAGES:
        lang = LANGUAGES[current_main_lang_key]
        st.session_state["lang"] = lang

    sidebar_lang_default = next((k for k, v in LANGUAGES.items() if v == lang), list(LANGUAGES.keys())[0])
    main_lang_key = st.sidebar.selectbox(
        TEXTS["main_language_label"][lang],
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(sidebar_lang_default),
        key="main_lang_selector"
    )
    lang = LANGUAGES[main_lang_key]
    st.session_state["lang"] = lang
    st.session_state["gate_lang_selector"] = main_lang_key

    st.title(TEXTS['app_title'][lang])
    st.markdown(f"### {TEXTS['main_section_adjustments_results'][lang]}")

    # --- REFERENCES SECTION WITH FULL METRICS SUITE ---
    ref_header_text = TEXTS['paper_ref'][lang]

    # BMB
    bmb_doi = "10.1007/s11538-026-01621-7"
    bmb_url = f"https://doi.org/{bmb_doi}"
    bmb_badge_img = "https://img.shields.io/badge/DOI-10.1007%2Fs11538--026--01621--7-blue.svg"
    
    # arXiv
    arxiv_doi = "10.48550/arXiv.2507.05960"
    
    # Zenodo
    zenodo_doi = "10.5281/zenodo.18025828"
    zenodo_url = f"https://doi.org/{zenodo_doi}"
    zenodo_badge_img = "https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18025828-blue.svg?logo=zenodo&logoColor=white"
    
    # Code Ocean
    code_ocean_doi = "10.24433/CO.0225069.v1" 
    code_ocean_url = f"https://doi.org/10.24433/CO.0225069.v1"
    code_ocean_badge_img = "https://img.shields.io/badge/Code_Ocean-Reproducible-blue.svg"
    
    badge_col_width = "210px"
    badge_min_height = "55px"
    
    badge_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: "Source Sans Pro", sans-serif; margin: 0; padding: 0; color: rgb(49, 51, 63); overflow: visible; width: 100%; }}
            .ref-header {{ font-size: 18px; font-weight: 700; margin-bottom: 20px; border-bottom: 1px solid #e6e6e6; padding-bottom: 5px; }}
            .rows-container {{ display: flex; flex-direction: column; gap: 25px; width: 100%; }}
            
            /* Layout em 2 colunas: alinha os itens no topo para os badges não flutuarem no meio do nada */
            .row {{ display: flex; align-items: flex-start; gap: 15px; width: 100%; }}
            
            .badge-wrapper {{ display: flex; align-items: center; gap: 8px; min-width: {badge_col_width}; min-height: {badge_min_height}; flex-shrink: 0; padding-top: 3px; }}
            
            /* Container principal para o texto + botões */
            .content-wrapper {{ display: flex; flex-direction: column; gap: 10px; flex: 1; }}
            
            .citation-text {{ font-family: 'Times New Roman', serif; font-size: 16px; line-height: 1.4; }}
            
            /* Flexbox horizontal para os botões. Wrap permite quebrar a linha se houver muitos */
            .link-badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
            .link-badges img {{ display: block; }}
        </style>
    </head>
    <body>
        <div class="ref-header">{ref_header_text}</div>
        <div class="rows-container">
            
            <div class="row">
                <div class="badge-wrapper">
                    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-doi='{bmb_doi}' data-hide-no-mentions='false'></div>
                    <a href="https://plu.mx/plum/a/?doi={bmb_doi}" class="plumx-plum-print-popup" data-popup="right" data-size="medium" data-pass-hidden-categories="true"></a>
                    <span class="__dimensions_badge_embed__" data-doi="{bmb_doi}" data-style="small_circle" data-hide-zero-citations="false"></span>
                </div>
                <div class="content-wrapper">
                    <div class="citation-text">Mockaitis, G. (2026) Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics. Bulletin of Mathematical Biology. 88:55. DOI: {bmb_doi}</div>
                    <div class="link-badges">
                        <a href="{bmb_url}" target="_blank"><img src="{bmb_badge_img}" alt="DOI"></a>
                        <a href="{bmb_url}" target="_blank"><img src="https://img.shields.io/badge/Open_Access-F68212.svg?logo=openaccess&logoColor=white" alt="Open Access"></a>
                        <a href="https://plu.mx/plum/a/?doi={bmb_doi}" target="_blank"><img src="https://img.shields.io/badge/PlumX-Metrics-7E2F8E.svg" alt="PlumX"></a>
                        <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
                    </div>
                </div>
            </div>
    
            <div class="row">
                <div class="badge-wrapper">
                    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-arxiv-id='2507.05960' data-hide-no-mentions='true'></div>
                    <a href="https://plu.mx/plum/a/?arxiv=2507.05960" class="plumx-plum-print-popup" data-popup="right" data-size="medium" data-pass-hidden-categories="true"></a>
                    <span class="__dimensions_badge_embed__" data-doi="{arxiv_doi}" data-style="small_circle" data-hide-zero-citations="false"></span>
                </div>
                <div class="content-wrapper">
                    <div class="citation-text">Mockaitis, G. (2025) Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics. ArXiv: 2507.05960, 42 p.</div>
                    <div class="link-badges">
                        <a href="https://doi.org/10.48550/arXiv.2507.05960" target="_blank"><img src="https://img.shields.io/badge/arXiv-2507.05960-b31b1b.svg" alt="arXiv"></a>
                        <a href="https://plu.mx/plum/a/?arxiv=2507.05960" target="_blank"><img src="https://img.shields.io/badge/PlumX-Metrics-7E2F8E.svg" alt="PlumX"></a>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="badge-wrapper">
                    <div class='altmetric-embed' data-badge-type='donut' data-badge-popover='right' data-doi='{zenodo_doi}' data-hide-no-mentions='false'></div>
                    <a href="https://plu.mx/plum/a/?doi={zenodo_doi}" class="plumx-plum-print-popup" data-popup="right" data-size="medium" data-pass-hidden-categories="true"></a>
                </div>
                <div class="content-wrapper">
                    <div class="citation-text">{TEXTS['zenodo_cite'][lang]}</div>
                    <div class="link-badges">
                        <a href="{zenodo_url}" target="_blank"><img src="{zenodo_badge_img}" alt="Zenodo DOI"></a>
                        <a href="{code_ocean_url}" target="_blank"><img src="{code_ocean_badge_img}" alt="Code Ocean DOI"></a>
                        <a href="https://plu.mx/plum/a/?doi={zenodo_doi}" target="_blank"><img src="https://img.shields.io/badge/PlumX-Metrics-7E2F8E.svg" alt="PlumX"></a>
                    </div>
                </div>
            </div>
            
        </div>
        
        <script type='text/javascript' src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>
        <script type="text/javascript" src="https://cdn.plu.mx/widget-popup.js"></script>
        <script async src="https://integration-badge.dimensions.ai/static/ai/badge.js" charset="utf-8"></script>
    </body>
    </html>
    """
    
    # Main Analysis Interface Sidebar
    st.sidebar.header(TEXTS['sidebar_config'][lang])
    param_labels = ("y_i", "y_f")
    rate_label = "r_max"
    default_x_label = TEXTS["axis_time"][lang]
    default_y_label = TEXTS["default_y_label"][lang]

    # Store settings changes to reset analysis automatically if parameter changes
    def reset_analysis():
        st.session_state.analysis_run = False

    seed_mode = st.sidebar.selectbox(
        TEXTS["seed_mode_label"][lang],
        options=["fixed_42", "random"],
        format_func=lambda x: TEXTS["seed_fixed_42"][lang] if x == "fixed_42" else TEXTS["seed_random"][lang],
        on_change=reset_analysis,
        key="seed_mode_selector"
    )

    file = st.sidebar.file_uploader(TEXTS['upload_label'][lang], type=["csv", "xlsx"], key="data_file_uploader")
    max_phases = st.sidebar.number_input(TEXTS['max_phases'][lang], 1, 10, 5, key="max_phases_input")

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

    outlier_method_key = st.sidebar.selectbox(
        TEXTS['outlier_method_label'][lang],
        options=outlier_options_keys,
        index=2,
        format_func=lambda k: TEXTS[f"outlier_{k}"][lang],
        on_change=reset_analysis,
        key="outlier_method_selector"
    )
    rout_q = 1.0
    if outlier_method_key == "rout":
        rout_q = st.sidebar.slider(
            TEXTS["rout_q_label"][lang],
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            on_change=reset_analysis,
            key="rout_q_slider"
        )
    
    # --- Constraints ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['constraints_header'][lang]}")
    
    force_yi = st.sidebar.checkbox(TEXTS['force_yi'][lang], value=False, on_change=reset_analysis, key="force_yi_checkbox")
    force_yf = st.sidebar.checkbox(TEXTS['force_yf'][lang], value=False, disabled=force_yi, on_change=reset_analysis, key="force_yf_checkbox")
    
    if force_yi:
        force_yf = False

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {TEXTS['graphic_options_header'][lang]}")
    selected_font = st.sidebar.selectbox(
        TEXTS["plot_font_label"][lang],
        options=PLOT_FONT_OPTIONS,
        index=0,
        on_change=reset_analysis,
        key="plot_font_selector"
    )
    apply_plot_font(selected_font)
    x_title_custom = st.sidebar.text_input(
        TEXTS["axis_x_custom_label"][lang],
        value="",
        on_change=reset_analysis,
        key="axis_x_custom"
    )
    y_title_custom = st.sidebar.text_input(
        TEXTS["axis_y_custom_label"][lang],
        value="",
        on_change=reset_analysis,
        key="axis_y_custom"
    )
    st.sidebar.caption(TEXTS["axis_math_hint"][lang])
    x_axis_label = normalize_axis_math_label(x_title_custom) if str(x_title_custom).strip() else default_x_label
    y_axis_label = normalize_axis_math_label(y_title_custom) if str(y_title_custom).strip() else default_y_label
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            save_uploaded_data(df, st.session_state.get("user_profile", {}))

            t_flat, y_flat, replicates = process_data(df)
            if not replicates:
                st.error(TEXTS['error_cols'][lang])
            else:
                # Always show Raw Data top graph
                plot_raw_data(replicates, lang, x_axis_label, y_axis_label)

                st.success(TEXTS['data_loaded'][lang].format(len(replicates), len(t_flat)))
                
                # Update Session State based on Button Click
                if st.button(TEXTS['run_button'][lang], key="run_analysis_button"):
                    if seed_mode == "fixed_42":
                        st.session_state.run_seed = 42
                    else:
                        st.session_state.run_seed = random.randint(0, 2_147_483_647)
                    st.session_state.analysis_run = True

                # ==========================================================
                # ANALYSIS EXECUTION BLOCK (Managed by Session State)
                # ==========================================================
                if st.session_state.analysis_run:
                    st.divider()
                    st.caption(TEXTS["seed_used_msg"][lang].format(st.session_state.run_seed))
                    
                    best_results_global = {"Gompertz": None, "Boltzmann": None}
                    first_order_single_phase_cache = None
                    
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
                                        res_standard = run_fit_with_outlier_strategy(
                                            t_flat,
                                            y_flat,
                                            func,
                                            n,
                                            outlier_method_key,
                                            force_yi=force_yi,
                                            force_yf=force_yf,
                                            rout_q=rout_q,
                                            use_first_order_phase1=False,
                                            random_seed=st.session_state.run_seed
                                        )

                                        if res_standard:
                                            st.markdown(
                                                f"**{TEXTS['fit_standard_header'][lang]}** - "
                                                f"`{get_full_model_label(res_standard, model_name, lang)}`"
                                            )
                                            display_single_fit(
                                                res_standard,
                                                replicates,
                                                model_name,
                                                func,
                                                color,
                                                x_axis_label,
                                                y_axis_label,
                                                param_labels,
                                                rate_label,
                                                lang
                                            )
                                            results_list.append(res_standard)
                                        else:
                                            st.warning(TEXTS['warning_insufficient'][lang])
                                            continue

                                        should_try_first_order = should_test_first_order_variant(res_standard, tol=1e-6)
                                        skip_duplicate_single_phase = (
                                            n == 1
                                            and model_name == "Boltzmann"
                                            and first_order_single_phase_cache is not None
                                        )
                                        if should_try_first_order and not skip_duplicate_single_phase:
                                            res_first_order = run_fit_with_outlier_strategy(
                                                t_flat,
                                                y_flat,
                                                func,
                                                n,
                                                outlier_method_key,
                                                force_yi=force_yi,
                                                force_yf=force_yf,
                                                rout_q=rout_q,
                                                use_first_order_phase1=True,
                                                random_seed=st.session_state.run_seed
                                            )
                                            if n == 1 and model_name == "Gompertz" and res_first_order is not None:
                                                first_order_single_phase_cache = {
                                                    "result": deepcopy(res_first_order),
                                                    "source_model": model_name
                                                }
                                            if res_first_order:
                                                st.markdown(
                                                    f"**{TEXTS['fit_first_order_header'][lang]}** - "
                                                    f"`{get_full_model_label(res_first_order, model_name, lang)}`"
                                                )
                                                display_single_fit(
                                                    res_first_order,
                                                    replicates,
                                                    model_name,
                                                    func,
                                                    color,
                                                    x_axis_label,
                                                    y_axis_label,
                                                    param_labels,
                                                    rate_label,
                                                    lang
                                                )
                                                results_list.append(res_first_order)
                                            else:
                                                st.warning(TEXTS['warning_insufficient'][lang])
                                        else:
                                            # Keep behavior explicit: no first-order additional fit when lambda_1 is not zero.
                                            pass

                            if results_list:
                                st.markdown(f"### {TEXTS['table_title'][lang]}")

                                N = len(y_flat)
                                k_values = [len(r['theta']) for r in results_list]
                                k_min, k_max = min(k_values), max(k_values)
                                ic_name = choose_information_criterion(N, k_max)
                                st.markdown(f"#### {TEXTS['selection_logic_title'][lang]}")
                                st.info(TEXTS["selection_logic_body"][lang].format(ic_name))

                                best_by_phase = get_best_candidate_by_phase(results_list, ic_name)
                                ic_values = [r['metrics'][ic_name] for r in best_by_phase]
                                best_idx = select_first_local_min_index(ic_values)
                                best_n = best_by_phase[best_idx]['n_phases']
                                best_results_global[model_name] = best_by_phase[best_idx]
                                best_final_full_label = get_full_model_label(best_by_phase[best_idx], model_name, lang)

                                summary_data = []
                                for r in results_list:
                                    m = r['metrics']
                                    summary_data.append({
                                        "F": r['n_phases'],
                                        TEXTS["full_model_col"][lang]: get_full_model_label(r, model_name, lang),
                                        TEXTS['metric_corr'][lang]: m.get('r', np.nan),
                                        "R²": m['R2'],
                                        "R² Adj": m['R2_adj'],
                                        "SSE": m['SSE'],
                                        "AIC": m['AIC'],
                                        "AICc": m['AICc'],
                                        "BIC": m['BIC'],
                                        TEXTS["phase1_col"][lang]: (
                                            TEXTS["phase1_first_order"][lang]
                                            if r.get("use_first_order_phase1", False)
                                            else TEXTS["phase1_sigmoidal"][lang]
                                        ),
                                        TEXTS['summary_header_used'][lang].format(ic_name): m[ic_name]
                                    })

                                summary_df = pd.DataFrame(summary_data)

                                best_phase_ic = {
                                    int(r["n_phases"]): float(r["metrics"][ic_name]) for r in best_by_phase
                                }
                                best_final_phase = int(best_n)
                                best_final_ic = float(best_by_phase[best_idx]["metrics"][ic_name])

                                def highlight_row(row):
                                    row_phase = int(row["F"])
                                    row_ic = float(row[TEXTS['summary_header_used'][lang].format(ic_name)])
                                    row_model = str(row[TEXTS["full_model_col"][lang]])
                                    is_phase_best = row_phase in best_phase_ic and abs(row_ic - best_phase_ic[row_phase]) <= 1e-12
                                    is_final = (
                                        row_phase == best_final_phase
                                        and abs(row_ic - best_final_ic) <= 1e-12
                                        and row_model == best_final_full_label
                                    )
                                    if is_final:
                                        return ['background-color: #d7e8ff; font-weight: bold; border: 2px solid #0d6efd'] * len(row)
                                    if is_phase_best:
                                        return ['background-color: #d4edda; font-weight: bold'] * len(row)
                                    return [''] * len(row)

                                st.dataframe(
                                    summary_df.style.apply(highlight_row, axis=1).format({
                                        TEXTS['metric_corr'][lang]: "{:.4f}",
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
                                st.caption(f"🟩 {TEXTS['legend_phase_best'][lang]}")
                                st.caption(f"🟦 {TEXTS['legend_final_best'][lang]}")

                                st.info(
                                    TEXTS['info_selection_criteria'][lang].format(ic_name, N, k_min, k_max, N / k_max, best_n)
                                )

                                st.success(
                                    TEXTS['best_model_msg'][lang].format(best_n, ic_name)
                                )

                                st.markdown(f"### {TEXTS['graph_summary_title'][lang]}")
                                plot_metrics_summary(best_by_phase, model_name, lang)

                    # --- FINAL SUMMARY GRAPH (Appears after tabs) ---
                    st.divider()
                    plot_final_summary(replicates, best_results_global, lang, x_axis_label, y_axis_label)
                    
                    # --- EXCEL EXPORT BUTTON ---
                    # Placed at the very end of the analysis so users can grab everything at once
                    st.divider()
                    excel_data = generate_excel_report(best_results_global, replicates, param_labels, rate_label, lang)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label=f"📊 {TEXTS['download_excel'][lang]}",
                            data=excel_data,
                            file_name=f"polyauxic_results_{datetime.now().strftime('%d-%m-%Y')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch"
                        )

        except Exception as e:
            st.error(TEXTS['error_proc'][lang].format(e))
    else:
        st.info(TEXTS['info_upload'][lang])

    st.markdown("---")
    st.info(TEXTS['intro_desc'][lang])
    with st.expander(TEXTS['instructions_header'][lang], expanded=False):
        st.markdown(TEXTS['instructions_list'][lang])
    components.html(badge_html, height=350, scrolling=False)

if __name__ == "__main__":
    main()
