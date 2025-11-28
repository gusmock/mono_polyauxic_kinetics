import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. DEFINIÇÕES MATEMÁTICAS (MODELOS E PERDA)
# ==============================================================================

def gompertz_step(t, A, mu, lam):
    """Modelo Gompertz Modificado (Eq. Zwietering)."""
    # np.clip evita overflow em exponenciais gigantes durante a busca
    exponent = np.clip(((mu * np.e) / A) * (lam - t) + 1, -100, 100)
    return A * np.exp(-np.exp(exponent))

def boltzmann_step(t, A, mu, lam):
    """Modelo Boltzmann Sigmoidal."""
    exponent = np.clip(((4 * mu) / A) * (lam - t) + 2, -100, 100)
    return A / (1 + np.exp(exponent))

def polyauxic_model(t, params, model_func, n_phases):
    """
    Soma de sigmoides para comportamento poliauxico.
    params: lista plana [A1, mu1, lam1, A2, mu2, lam2...]
    """
    y_sum = 0
    # Itera sobre trios de parâmetros
    for i in range(n_phases):
        idx = i * 3
        if idx + 2 < len(params):
            A = params[idx]
            mu = params[idx+1]
            lam = params[idx+2]
            y_sum += model_func(t, A, mu, lam)
    return y_sum

def lorentzian_loss(params, t_exp, y_exp, model_func, n_phases):
    """
    Função de Perda Lorentziana (Robust Loss).
    Penaliza outliers menos severamente que os Mínimos Quadrados (SSE).
    """
    y_pred = polyauxic_model(t_exp, params, model_func, n_phases)
    residuals = y_exp - y_pred
    
    # Escala robusta (MAD - Median Absolute Deviation)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = mad if mad > 1e-6 else 1.0 # Evitar divisão por zero
    
    # Perda Lorentziana: log(1 + (erro/escala)^2)
    loss = np.sum(np.log(1 + (residuals / scale)**2))
    return loss

def calculate_statistics(y_true, y_pred, n_params):
    """Calcula R2, AIC e BIC para seleção de modelo."""
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    
    # R-squared
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0
    
    # AIC e BIC (Assumindo erro normal para simplificação da verossimilhança)
    if sse > 0:
        log_likelihood = -n/2 * (np.log(2 * np.pi * sse/n) + 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
    else:
        aic, bic = np.inf, np.inf
        
    return r2, aic, bic

# ==============================================================================
# 2. INTERFACE E LÓGICA DO APP
# ==============================================================================

def main():
    st.set_page_config(page_title="Ajuste Cinético Poliauxico", layout="wide")
    st.title("🧬 Ajuste de Cinética: Algoritmo Híbrido (Global + Local)")
    
    st.markdown("""
    **Metodologia baseada no artigo:**
    1. Limpeza de Dados (Tratamento numérico).
    2. Busca Global (Differential Evolution - similar ao PSO).
    3. Refinamento Local (Nelder-Mead).
    4. Perda Lorentziana (Robusta a outliers).
    """)

    # --- 1. ENTRADA DE DADOS ---
    st.sidebar.header("1. Carregar Dados")
    uploaded_file = st.sidebar.file_uploader("Arquivo Excel ou CSV", type=["xlsx", "csv"])

    # Inicialização segura
    t_data, y_data = None, None

    if uploaded_file:
        try:
            # Leitura
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Seleção de Colunas
            cols = df.columns.tolist()
            c1, c2 = st.sidebar.columns(2)
            t_col = c1.selectbox("Tempo (t)", cols, index=0)
            y_col = c2.selectbox("Biomassa/Prod (y)", cols, index=1 if len(cols)>1 else 0)

            # LIMPEZA ROBUSTA (Corrigindo erro de vírgula/ponto)
            # Converte tudo para string, troca vírgula por ponto, converte para float
            t_clean = pd.to_numeric(df[t_col].astype(str).str.replace(',', '.'), errors='coerce')
            y_clean = pd.to_numeric(df[y_col].astype(str).str.replace(',', '.'), errors='coerce')

            # Remove NaNs
            mask = ~np.isnan(t_clean) & ~np.isnan(y_clean)
            t_data = t_clean[mask].values
            y_data = y_clean[mask].values

            # Ordena pelo tempo
            idx_sort = np.argsort(t_data)
            t_data = t_data[idx_sort]
            y_data = y_data[idx_sort]

            if len(t_data) < 5:
                st.error("Erro: Poucos dados válidos encontrados. Verifique a formatação.")
                st.stop()
            
            # Scatter Plot Inicial
            st.sidebar.markdown("---")
            st.sidebar.write(f"✅ **{len(t_data)} pontos** processados.")

        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

    # Se não houver dados, para aqui
    if t_data is None:
        st.info("👈 Por favor, carregue um arquivo na barra lateral.")
        st.stop()

    # --- 2. CONFIGURAÇÃO DO MODELO ---
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        model_choice = st.radio("Escolha o Modelo:", ["Gompertz", "Boltzmann"])
        model_func = gompertz_step if model_choice == "Gompertz" else boltzmann_step
    
    with col_conf2:
        n_phases = st.number_input("Número de Fases (Poliauxia)", 1, 4, 1)

    # --- 3. EXECUÇÃO DO ALGORITMO ---
    if st.button("🚀 EXECUTAR AJUSTE HÍBRIDO (Global + Local)"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # --- FASE 1: DEFINIR LIMITES (BOUNDS) PARA BUSCA GLOBAL ---
            status_text.text("1/3 Configurando Espaço de Busca...")
            
            # Limites automáticos baseados nos dados
            y_max = np.max(y_data)
            t_max = np.max(t_data)
            
            # Bounds: [(min, max), (min, max)...] para cada parâmetro
            # A (0 a 1.5x max), mu (0 a 5.0), lambda (0 a t_max)
            bounds = []
            for _ in range(n_phases):
                bounds.append((0, y_max * 1.5))   # A
                bounds.append((0, (y_max/t_max)*10)) # mu (estimativa larga)
                bounds.append((0, t_max))         # lambda
            
            # --- FASE 2: BUSCA GLOBAL (Differential Evolution) ---
            # Isso substitui o PSO do artigo com robustez similar/superior no Scipy
            status_text.text("2/3 Rodando Otimização Global (Pode levar alguns segundos)...")
            progress_bar.progress(30)
            
            global_res = differential_evolution(
                lorentzian_loss, 
                bounds, 
                args=(t_data, y_data, model_func, n_phases),
                strategy='best1bin', 
                maxiter=1000, 
                popsize=15, 
                tol=0.01,
                seed=42 # Reprodutibilidade
            )
            
            params_global = global_res.x
            
            # --- FASE 3: REFINAMENTO LOCAL (Nelder-Mead) ---
            status_text.text("3/3 Refinamento Local (Nelder-Mead)...")
            progress_bar.progress(70)
            
            local_res = minimize(
                lorentzian_loss,
                params_global, # Começa de onde o Global parou
                args=(t_data, y_data, model_func, n_phases),
                method='Nelder-Mead',
                tol=1e-6
            )
            
            final_params = local_res.x
            progress_bar.progress(100)
            status_text.text("Concluído!")
            
            # --- 4. RESULTADOS E ESTATÍSTICAS ---
            st.divider()
            
            # Curva Ajustada
            t_smooth = np.linspace(min(t_data), max(t_data), 300)
            y_smooth = polyauxic_model(t_smooth, final_params, model_func, n_phases)
            y_pred_metrics = polyauxic_model(t_data, final_params, model_func, n_phases)
            
            # Calcular AIC/BIC
            r2, aic, bic = calculate_statistics(y_data, y_pred_metrics, len(final_params))
            
            # Layout de Resultados
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.subheader("Parâmetros Cinéticos")
                
                stats_df = pd.DataFrame({
                    "Métrica": ["R² (Ajuste)", "AIC", "BIC", "Loss (Lorentz)"],
                    "Valor": [f"{r2:.4f}", f"{aic:.2f}", f"{bic:.2f}", f"{local_res.fun:.4f}"]
                })
                st.table(stats_df)
                
                # Tabela de Parâmetros por Fase
                res_list = []
                for i in range(n_phases):
                    idx = i * 3
                    res_list.append({
                        "Fase": i+1,
                        "A (Assíntota)": f"{final_params[idx]:.4f}",
                        "μ (Taxa Máx)": f"{final_params[idx+1]:.4f}",
                        "λ (Fase Lag)": f"{final_params[idx+2]:.4f}"
                    })
                st.table(pd.DataFrame(res_list))
                
            with col_res2:
                st.subheader("Gráfico do Modelo")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Dados Reais
                ax.scatter(t_data, y_data, color='black', alpha=0.6, label='Experimental', s=30)
                
                # Curva Total
                ax.plot(t_smooth, y_smooth, color='red', linewidth=2, label='Modelo Global Ajustado')
                
                # Desenhar as fases individuais (se for poliauxico)
                if n_phases > 1:
                    colors = ['blue', 'green', 'orange', 'purple']
                    for i in range(n_phases):
                        idx = i * 3
                        p_phase = final_params[idx:idx+3]
                        # Simula apenas esta fase (nota: é ilustrativo, pois as fases se somam)
                        y_phase_viz = model_func(t_smooth, *p_phase) 
                        ax.plot(t_smooth, y_phase_viz, linestyle='--', alpha=0.5, 
                                color=colors[i % len(colors)], label=f'Fase {i+1} Isolada')

                ax.set_xlabel("Tempo (h)")
                ax.set_ylabel("Biomassa / Produto (g/L)")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocorreu um erro matemático durante o ajuste: {e}")
            st.error("Dica: Tente reduzir o número de fases ou verificar se os dados têm formato coerente.")

if __name__ == "__main__":
    main()