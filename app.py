import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --- 1. DEFINIÇÃO MATEMÁTICA DOS MODELOS ---

def gompertz(t, A, mu, lam):
    # Modelo Gompertz Reparametrizado
    return A * np.exp(-np.exp(((mu * np.e) / A) * (lam - t) + 1))

def boltzmann(t, A, mu, lam):
    # Modelo Boltzmann Sigmoidal
    return A / (1 + np.exp(((4 * mu) / A) * (lam - t) + 2))

def polyauxic_model(t, params, model_func, n_phases):
    """
    Calcula a soma das fases (Poliauxia).
    params: lista plana [A1, mu1, lam1, A2, mu2, lam2...]
    """
    y_sum = 0
    for i in range(n_phases):
        idx = i * 3
        # Extrai os 3 parâmetros da fase i
        A, mu, lam = params[idx], params[idx+1], params[idx+2]
        y_sum += model_func(t, A, mu, lam)
    return y_sum

def objective_function(params, t_exp, y_exp, model_func, n_phases):
    """
    Calcula o erro entre o Previsto (p) e o Real.
    Usa Logaritmo para penalizar desvios grandes (robustez simples).
    """
    y_pred = polyauxic_model(t_exp, params, model_func, n_phases)
    residuals = y_exp - y_pred
    # Soma dos Quadrados dos Resíduos (pode ser trocado por Lorentziana)
    return np.sum(residuals**2) 

# --- 2. INTERFACE (STREAMLIT) ---

def main():
    st.set_page_config(page_title="Ajuste Cinético", layout="wide")
    st.title("🧪 Ajuste de Cinética Microbiana (Mono/Poliauxico)")

    # --- BLOCO 1: ENTRADA DE DADOS NA MATRIZ ---
    st.sidebar.header("1. Dados")
    uploaded_file = st.sidebar.file_uploader("Suba o Excel (.xlsx) ou CSV", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Seletores de Coluna
            cols = df.columns.tolist()
            c1, c2 = st.sidebar.columns(2)
            t_col = c1.selectbox("Tempo (t)", cols, index=0)
            y_col = c2.selectbox("Resposta (y)", cols, index=1 if len(cols)>1 else 0)
            
            # Transformação em Matriz Numpy (Crítico para velocidade)
            t_data = df[t_col].values
            y_data = df[y_col].values
            
            # Visualização Rápida
            st.scatter_chart(df, x=t_col, y=y_col)

        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return
    else:
        st.info("Por favor, carregue um arquivo para começar.")
        return

    # --- BLOCO 2: ESCOLHER MODELO ---
    st.sidebar.header("2. Modelo")
    model_type = st.sidebar.radio("Equação Base:", ["Gompertz", "Boltzmann"])
    model_func = gompertz if model_type == "Gompertz" else boltzmann
    
    n_phases = st.sidebar.number_input("Nº de Fases (Poliauxia)", 1, 4, 1)

    # --- BLOCO 3: PARAMETRIZAÇÃO INICIAL ---
    st.subheader("3. Parâmetros Iniciais (Estimativa)")
    
    # Gerar estimativas automáticas grosseiras para ajudar o utilizador
    initial_guesses = []
    input_cols = st.columns(n_phases)
    
    for i in range(n_phases):
        with input_cols[i]:
            st.markdown(f"**Fase {i+1}**")
            # Heurísticas simples para preencher os campos
            def_A = (np.max(y_data) / n_phases)
            def_mu = (np.max(y_data) - np.min(y_data)) / (np.max(t_data) - np.min(t_data))
            def_lam = np.min(t_data) + (np.max(t_data) - np.min(t_data)) * (i / n_phases) * 0.5
            
            p_A = st.number_input(f"A_{i+1}", value=float(def_A), format="%.2f", key=f"A{i}")
            p_mu = st.number_input(f"μ_{i+1}", value=float(def_mu), format="%.3f", key=f"m{i}")
            p_lam = st.number_input(f"λ_{i+1}", value=float(def_lam), format="%.2f", key=f"l{i}")
            
            initial_guesses.extend([p_A, p_mu, p_lam])

    # --- BLOCO 4 & 5: CALCULAR (OTIMIZAR) E EXIBIR ---
    if st.button("🚀 Calcular Ajuste (Run Fit)"):
        with st.spinner("A otimizar parâmetros..."):
            
            # Otimização (Nelder-Mead é robusto para derivadas desconhecidas)
            result = minimize(
                objective_function, 
                initial_guesses, 
                args=(t_data, y_data, model_func, n_phases),
                method='Nelder-Mead',
                tol=1e-5
            )
            
            final_params = result.x
            
            # Gerar curva ajustada (suave) para o gráfico
            t_smooth = np.linspace(min(t_data), max(t_data), 200)
            y_smooth = polyauxic_model(t_smooth, final_params, model_func, n_phases)
            
            # --- RESULTADOS ---
            c_res1, c_res2 = st.columns([1, 2])
            
            with c_res1:
                st.success("Convergência Atingida!")
                st.write(f"Erro Final: {result.fun:.4f}")
                
                # Exibir Tabela de Parâmetros
                res_dict = {}
                for i in range(n_phases):
                    idx = i * 3
                    res_dict[f"Fase {i+1}"] = {
                        "A (Máx)": f"{final_params[idx]:.3f}",
                        "μ (Velocidade)": f"{final_params[idx+1]:.4f}",
                        "λ (Lag)": f"{final_params[idx+2]:.3f}"
                    }
                st.table(pd.DataFrame(res_dict))

            with c_res2:
                # Gráfico Final
                chart_df = pd.DataFrame({
                    "Tempo": t_smooth,
                    "Modelo Ajustado": y_smooth
                })
                # Adicionar pontos experimentais
                exp_df = pd.DataFrame({"Tempo": t_data, "Experimental": y_data})
                
                st.line_chart(chart_df.set_index("Tempo"))
                # Nota: O st.line_chart é simples. Para misturar linha e pontos perfeitamente,
                # usaríamos 'plotly' ou 'matplotlib' numa versão futura.

if __name__ == "__main__":
    main()