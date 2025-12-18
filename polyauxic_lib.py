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

"""
import streamlit as st
import pandas as pd
import numpy as np
import polyauxic_lib as lib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ================= CONFIGURAÇÃO =================
st.set_page_config(page_title="Experimento de Estabilidade (Seeds)", layout="wide")

st.title("🧪 Análise de Estabilidade Numérica (Seeds)")
st.markdown("""
Esta ferramenta avalia a robustez dos modelos variando a **Semente Aleatória (Seed)**.
* **FDR:** Fixado em 1.0%
* **Seeds:** 5 sementes diferentes por condição
* **Objetivo:** Gerar superfície de resposta (Heatmap) da variação do AICc.
""")

# ================= SIDEBAR =================
st.sidebar.header("Configurações")
uploaded_files = st.sidebar.file_uploader(
    "Carregar Datasets (CSV/XLSX)", 
    accept_multiple_files=True, 
    type=["csv", "xlsx"]
)

# SEEDS CONFIG
SEEDS = [42, 123, 777, 2024, 9999]
FIXED_FDR = 1.0
st.sidebar.info(f"FDR fixado em {FIXED_FDR}%")
st.sidebar.info(f"Seeds: {SEEDS}")

# MODELOS
st.sidebar.subheader("Modelos")
use_gompertz = st.sidebar.checkbox("Gompertz", True)
use_boltzmann = st.sidebar.checkbox("Boltzmann", True)

# RESTRIÇÕES
st.sidebar.subheader("Restrições")
use_floating = st.sidebar.checkbox("Floating (Livre)", True)
use_forced = st.sidebar.checkbox("Forced (yi=0)", True)

# ================= EXECUÇÃO =================
if st.button("🚀 INICIAR EXPERIMENTO", type="primary"):
    if not uploaded_files:
        st.error("Por favor, carregue pelo menos um arquivo.")
        st.stop()
        
    MODELS = []
    if use_gompertz: MODELS.append(("Gompertz", lib.gompertz_term_eq32))
    if use_boltzmann: MODELS.append(("Boltzmann", lib.boltzmann_term_eq31))
    
    CONSTRAINTS = []
    if use_floating: CONSTRAINTS.append(False)
    if use_forced: CONSTRAINTS.append(True)
    
    if not MODELS or not CONSTRAINTS:
        st.error("Selecione pelo menos um modelo e uma restrição.")
        st.stop()

    # Barra de Progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(uploaded_files) * len(MODELS) * len(CONSTRAINTS) * len(SEEDS)
    step = 0
    all_results = []
    
    # LOOP PRINCIPAL
    for file_obj in uploaded_files:
        filename = file_obj.name
        
        # Identificar Classe
        lower = filename.lower()
        if "replicates" in lower: c_type = "Replicates"
        elif "1storder" in lower: c_type = "First_Order"
        elif "unfinished" in lower: c_type = "Unfinished"
        else: c_type = "Outros"
        
        # Ler Dados
        try:
            if filename.endswith(".xlsx"): df = pd.read_excel(file_obj)
            else: df = pd.read_csv(file_obj)
            t_flat, y_flat, _ = lib.process_data(df)
            if len(t_flat) == 0: continue
        except Exception as e:
            st.warning(f"Erro ao ler {filename}: {e}")
            continue
            
        for model_name, model_func in MODELS:
            for force_yi in CONSTRAINTS:
                for seed in SEEDS:
                    step += 1
                    progress_bar.progress(min(step / total_steps, 1.0))
                    status_text.text(f"Processando: {filename} | {model_name} | Seed {seed}")
                    
                    # OTIMIZAÇÃO (Busca Melhor AICc entre 1-5 fases)
                    best_res = None
                    best_val = np.inf
                    
                    for n in range(1, 6):
                        # 1. Outliers
                        res_pre = lib.fit_model_auto_robust_pre(t_flat, y_flat, model_func, n, force_yi, False, seed)
                        if res_pre:
                            mask = lib.detect_outliers_rout_rigorous(y_flat, res_pre['y_pred'], Q=FIXED_FDR)
                            t_c = t_flat[~mask] if np.any(mask) else t_flat
                            y_c = y_flat[~mask] if np.any(mask) else y_flat
                            
                            # 2. Fit Final
                            res = lib.fit_model_auto(t_c, y_c, model_func, n, force_yi, False, seed)
                            if res and res['metrics']['AICc'] < best_val:
                                best_val = res['metrics']['AICc']
                                best_res = res
                                best_res['outliers'] = np.sum(mask)

                    if best_res:
                        all_results.append({
                            "Dataset": filename,
                            "Class": c_type,
                            "Model": model_name,
                            "Constraint": "Forced" if force_yi else "Floating",
                            "Seed": seed,
                            "AICc": best_res['metrics']['AICc'],
                            "SSE": best_res['metrics']['SSE'],
                            "Outliers": best_res['outliers']
                        })
    
    progress_bar.empty()
    status_text.success("Experimento Concluído!")
    
    # ================= RESULTADOS =================
    if all_results:
        df_res = pd.DataFrame(all_results)
        
        # 1. Tabela
        st.subheader("Resultados Brutos")
        st.dataframe(df_res)
        
        # Download CSV
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV Completo", csv, "seed_results.csv", "text/csv")
        
        # 2. HEATMAP (Superfície de Resposta)
        st.subheader("Superfície de Resposta: Estabilidade (Desvio Padrão do AICc)")
        
        stability = df_res.groupby(['Dataset', 'Model', 'Constraint'])['AICc'].std().reset_index()
        stability.rename(columns={'AICc': 'StdDev_AICc'}, inplace=True)
        stability['Config'] = stability['Model'] + " (" + stability['Constraint'] + ")"
        
        heatmap_data = stability.pivot(index="Dataset", columns="Config", values="StdDev_AICc")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", ax=ax)
        plt.title(f"Instabilidade do Modelo (StdDev entre {len(SEEDS)} seeds)")
        plt.ylabel("Dataset")
        plt.xlabel("Configuração do Modelo")
        st.pyplot(fig)
        
    else:
        st.warning("Nenhum resultado gerado.")
