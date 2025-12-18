import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import polyauxic_lib as lib 

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Aviso: 'seaborn' não instalado. O Heatmap será simples.")

# ================= CONFIGURAÇÃO =================
DATASET_FOLDER = "datasets/"
OUTPUT_FILE = "seed_experiment_results.csv"
PLOT_FILE = "heatmap_stability.png"

# FATORES EXPERIMENTAIS
# 1. Modelo
MODELS = [("Gompertz", lib.gompertz_term_eq32), ("Boltzmann", lib.boltzmann_term_eq31)]
# 2. Restrição
CONSTRAINTS = [False, True]  # False=Floating, True=Forced
# 3. Seed (Aleatoriedade)
SEEDS = [42, 123, 777, 2024, 9999] 
# 4. FDR (FIXO)
FIXED_FDR_Q = 1.0  # <--- FDR FIXADO EM 1%

# ================= EXECUÇÃO =================
def run_batch():
    all_results = []
    
    if not os.path.exists(DATASET_FOLDER):
        print(f"Erro: Pasta '{DATASET_FOLDER}' não encontrada.")
        return

    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv") or f.endswith(".xlsx")]
    print(f"Encontrados {len(files)} datasets. Iniciando Experimento com FDR={FIXED_FDR_Q}...")
    
    for filename in files:
        filepath = os.path.join(DATASET_FOLDER, filename)
        
        # Identificar Classe
        lower = filename.lower()
        if "replicates" in lower: c_type = "Replicates"
        elif "1storder" in lower: c_type = "First_Order"
        elif "unfinished" in lower: c_type = "Unfinished"
        else: c_type = "Unknown"

        print(f"  Processando {filename}...")
        
        # Carregar Dados
        try:
            if filename.endswith(".xlsx"): df = pd.read_excel(filepath)
            else: df = pd.read_csv(filepath)
            t_flat, y_flat, _ = lib.process_data(df)
            if len(t_flat) == 0: continue
        except Exception as e:
            print(f"    Erro ao ler {filename}: {e}")
            continue

        # Loop dos Fatores
        for model_name, model_func in MODELS:
            for force_yi in CONSTRAINTS:
                for seed in SEEDS:
                    
                    # Para cada seed, ajustamos fases de 1 a 5 e pegamos o MELHOR AICc (Parsimony 'Best')
                    best_fit = None
                    best_aicc = np.inf
                    
                    for n in range(1, 6):
                        # 1. Deteção de Outliers (Pré-ajuste Robusto com Seed)
                        res_pre = lib.fit_model_auto_robust_pre(t_flat, y_flat, model_func, n, force_yi, False, seed)
                        
                        if res_pre:
                            # Aplica ROUT com FDR=1.0 Fixo
                            mask = lib.detect_outliers_rout_rigorous(y_flat, res_pre['y_pred'], Q=FIXED_FDR_Q)
                            
                            if np.any(mask):
                                t_c = t_flat[~mask]
                                y_c = y_flat[~mask]
                            else:
                                t_c, y_c = t_flat, y_flat
                                
                            # 2. Ajuste Final (SSE)
                            res = lib.fit_model_auto(t_c, y_c, model_func, n, force_yi, False, seed)
                            
                            # Seleção simples: Menor AICc vence
                            if res and res['metrics']['AICc'] < best_aicc:
                                best_aicc = res['metrics']['AICc']
                                best_fit = res
                                best_fit['outliers_count'] = np.sum(mask)

                    if best_fit:
                        all_results.append({
                            "Dataset": filename,
                            "Class": c_type,
                            "Model": model_name,
                            "Constraint": "Forced" if force_yi else "Floating",
                            "Seed": seed,
                            "Selected_Phases": best_fit['n_phases'],
                            "AICc": best_fit['metrics']['AICc'],
                            "SSE": best_fit['metrics']['SSE'],
                            "Outliers": best_fit['outliers_count']
                        })

    # Salvar Resultados
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Resultados salvos em {OUTPUT_FILE}")

    # ================= GERAR HEATMAP =================
    if df_res.empty: return

    print("Gerando Heatmap de Estabilidade...")
    
    # Métrica de Estabilidade: Desvio Padrão do AICc entre as seeds
    # Agrupamos por Dataset e Configuração (Modelo + Restrição)
    stability = df_res.groupby(['Dataset', 'Model', 'Constraint'])['AICc'].std().reset_index()
    stability.rename(columns={'AICc': 'AICc_StdDev'}, inplace=True)
    
    # Criar rótulo composto para o eixo X
    stability['Config'] = stability['Model'] + " (" + stability['Constraint'] + ")"
    
    # Formato Matriz para Heatmap
    heatmap_data = stability.pivot(index="Dataset", columns="Config", values="AICc_StdDev")
    
    plt.figure(figsize=(12, 10))
    
    if HAS_SEABORN:
        # Heatmap bonito com Seaborn
        sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", 
                    cbar_kws={'label': 'Desvio Padrão AICc (Instabilidade)'})
    else:
        # Fallback simples Matplotlib
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Desvio Padrão AICc')
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45, ha='right')
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.title(f"Estabilidade Numérica (FDR={FIXED_FDR_Q}%) - Variação entre {len(SEEDS)} Seeds")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"📊 Gráfico salvo em {PLOT_FILE}")

if __name__ == "__main__":
    run_batch()
