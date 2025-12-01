import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. MODELOS SIGMOIDAIS (versões reparametrizadas simples)
# ==============================================================================

def gompertz_step(t, A, mu, lam):
    """
    Gompertz modificado:
    A  = assíntota (amplitude)
    mu = taxa máxima aparente
    lam = lag
    """
    exponent = np.clip(((mu * np.e) / A) * (lam - t) + 1, -100, 100)
    return A * np.exp(-np.exp(exponent))


def boltzmann_step(t, A, mu, lam):
    """
    Boltzmann sigmoidal:
    A  = assíntota (amplitude)
    mu = taxa máxima aparente
    lam = "centro" da transição (~lag)
    """
    exponent = np.clip(((4 * mu) / A) * (lam - t) + 2, -100, 100)
    return A / (1 + np.exp(exponent))


def polyauxic_model(t, params, model_func, n_phases):
    """
    Soma de sigmoides (modelo poliauxico).
    params = [A1, mu1, lam1, A2, mu2, lam2, ...]
    """
    y_sum = 0.0
    for i in range(n_phases):
        idx = i * 3
        A = params[idx]
        mu = params[idx + 1]
        lam = params[idx + 2]
        y_sum += model_func(t, A, mu, lam)
    return y_sum


# ==============================================================================
# 2. FUNÇÃO DE PERDA, ESTATÍSTICAS E CRITÉRIOS DE INFORMAÇÃO
# ==============================================================================

def lorentzian_loss(params, t_exp, y_exp, model_func, n_phases):
    """
    Perda Lorentziana robusta (como no artigo).
    """
    y_pred = polyauxic_model(t_exp, params, model_func, n_phases)
    residuals = y_exp - y_pred

    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = mad if mad > 1e-6 else 1.0

    return np.sum(np.log(1.0 + (residuals / scale) ** 2))


def calculate_sse_r2(y_true, y_pred):
    residuals = y_true - y_pred
    sse = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
    return sse, r2


def information_criteria(y_true, y_pred, n_params):
    """
    Calcula AIC, AICc e BIC assumindo erro normal (como no artigo).
    """
    n_data = len(y_true)
    sse, _ = calculate_sse_r2(y_true, y_pred)

    if sse <= 0:
        # proteção numérica
        return np.inf, np.inf, np.inf, sse

    # log-verossimilhança para erro normal de variância constante
    logL = -n_data / 2 * (np.log(2 * np.pi * sse / n_data) + 1)

    AIC = 2 * n_params - 2 * logL

    # AICc só faz sentido se n_data - n_params - 1 > 0
    if n_data - n_params - 1 > 0:
        AICc = AIC + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
    else:
        AICc = np.inf

    BIC = n_params * np.log(n_data) - 2 * logL

    return AIC, AICc, BIC, sse


def select_ic_value(n_data, n_params, AIC, AICc, BIC):
    """
    Implementa a lógica do fluxograma:
        - se N > 200 -> usa BIC
        - se N <= 200 e N/k < 40 -> usa AICc
        - senão -> usa AIC

    Atenção à notação do artigo:
        N = número de pontos (n_data)
        k = número de parâmetros (n_params)
    """
    if n_data > 200:
        return BIC, "BIC"
    else:
        # N/k < 40  <=>  n_data / n_params < 40
        if n_data / n_params < 40:
            return AICc, "AICc"
        else:
            return AIC, "AIC"


# ==============================================================================
# 3. HESSIANA NUMÉRICA E ERROS PADRÃO
# ==============================================================================

def numerical_hessian(func, x, eps=1e-4):
    """
    Hessiana numérica simples (central finite differences).
    func: função de uma variável vetorial x -> escalar
    x: array 1D
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    # base vectors
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps

            fpp = func(x + ei + ej)
            fpm = func(x + ei - ej)
            fmp = func(x - ei + ej)
            fmm = func(x - ei - ej)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)

    return H


def parameter_uncertainty(theta_hat, t_data, y_data, model_func, n_phases):
    """
    Estima erros padrão dos parâmetros:
        - Hessiana da LOSS de Lorentz
        - Covariância ~ sigma^2 * H^{-1}
        - sigma^2 estimado via SSE/(N - k)
    """
    n_data = len(y_data)
    n_params = len(theta_hat)

    # SSE com o modelo final
    y_pred = polyauxic_model(t_data, theta_hat, model_func, n_phases)
    sse, _ = calculate_sse_r2(y_data, y_pred)

    if n_data <= n_params:
        return np.full_like(theta_hat, np.nan)

    sigma2 = sse / (n_data - n_params)

    def loss_for_hessian(p):
        return lorentzian_loss(p, t_data, y_data, model_func, n_phases)

    try:
        H = numerical_hessian(loss_for_hessian, theta_hat)
        C = sigma2 * np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(C)))
    except Exception:
        se = np.full_like(theta_hat, np.nan)

    return se


# ==============================================================================
# 4. AJUSTE AUTOMÁTICO PARA UM MODELO DADO (Gompertz OU Boltzmann)
# ==============================================================================

def fit_model_family(t_data, y_data, model_func, model_name, max_phases=4,
                     de_maxiter=500, random_seed=42):

    """
    Retorna o melhor ajuste para um modelo (Gompertz ou Boltzmann),
    varrendo n_fases = 1..max_phases e usando AIC/AICc/BIC conforme o artigo.
    """

    rng = np.random.default_rng(random_seed)

    n_data = len(y_data)
    y_max = np.max(y_data)
    t_max = np.max(t_data)

    best_result = None
    Cr = np.inf   # custo de referência (melhor critério até agora)

    for n_phases in range(1, max_phases + 1):

        # -----------------------------
        # 4.1 Bounds dos parâmetros
        # -----------------------------
        bounds = []
        for _ in range(n_phases):
            # A: assíntota ~ 0 -> 1.5 * y_max
            bounds.append((0, 1.5 * y_max))
            # mu: taxa máxima ~ 0 -> (y_max / t_max) * 10  (chute largo)
            bounds.append((0, (y_max / max(t_max, 1e-6)) * 10))
            # lam: lag entre 0 e t_max
            bounds.append((0, t_max))

        # -----------------------------
        # 4.2 Otimização global (DE)
        # -----------------------------
        global_res = differential_evolution(
            lorentzian_loss,
            bounds,
            args=(t_data, y_data, model_func, n_phases),
            strategy='best1bin',
            maxiter=de_maxiter,
            popsize=15,
            tol=0.01,
            seed=int(rng.integers(0, 1_000_000)),
            polish=False  # refinamento virá depois via Nelder-Mead
        )

        theta_global = global_res.x

        # -----------------------------
        # 4.3 Refinamento local (Nelder–Mead)
        # -----------------------------
        local_res = minimize(
            lorentzian_loss,
            theta_global,
            args=(t_data, y_data, model_func, n_phases),
            method='Nelder-Mead',
            tol=1e-6
        )

        theta_hat = local_res.x
        loss_val = local_res.fun

        # -----------------------------
        # 4.4 Métricas e critérios de informação
        # -----------------------------
        y_pred = polyauxic_model(t_data, theta_hat, model_func, n_phases)
        AIC, AICc, BIC, sse = information_criteria(y_data, y_pred, len(theta_hat))
        ic_value, ic_name = select_ic_value(n_data, len(theta_hat), AIC, AICc, BIC)
        _, r2 = calculate_sse_r2(y_data, y_pred)

        # -----------------------------
        # 4.5 Atualiza melhor modelo se necessário (Cc < Cr)
        # -----------------------------
        Cc = ic_value
        if Cc < Cr:
            Cr = Cc
            se_params = parameter_uncertainty(theta_hat, t_data, y_data, model_func, n_phases)

            best_result = {
                "model_name": model_name,
                "model_func": model_func,
                "n_phases": n_phases,
                "theta_hat": theta_hat,
                "theta_se": se_params,
                "loss": loss_val,
                "AIC": AIC,
                "AICc": AICc,
                "BIC": BIC,
                "ic_used": ic_name,
                "ic_value": ic_value,
                "sse": sse,
                "r2": r2
            }
        else:
            # critério de parada: se piorou, paramos de aumentar n
            break

    return best_result


# ==============================================================================
# 5. INTERFACE STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Cinética Mono/Poliauxica (Gompertz x Boltzmann)",
                       layout="wide")
    st.title("🧬 Ajuste Automático Mono/Poliauxico (Gompertz x Boltzmann)")

    st.markdown("""
Este app:

1. Lê dados de tempo × resposta;
2. Roda **Gompertz** e **Boltzmann**;
3. Para cada modelo, procura o melhor número de fases (1..Nₘₐₓ)
   usando **Lorentzian loss + Differential Evolution + Nelder–Mead**;
4. Usa **AIC / AICc / BIC** para decidir quantas fases usar em cada modelo;
5. Compara o melhor Gompertz vs melhor Boltzmann (mesmo critério) e escolhe o vencedor.
""")

    # -----------------------------
    # 5.1 Entrada de dados
    # -----------------------------
    st.sidebar.header("1. Dados de entrada")
    uploaded_file = st.sidebar.file_uploader("Arquivo CSV ou Excel",
                                             type=["csv", "xlsx"])

    max_phases = st.sidebar.number_input("Nº máximo de fases (Nₘₐₓ)",
                                         min_value=1, max_value=6, value=4, step=1)
    de_maxiter = st.sidebar.number_input("Maxiter (Differential Evolution)",
                                         min_value=50, max_value=2000, value=500, step=50)
    random_seed = st.sidebar.number_input("Seed aleatória", value=42, step=1)

    if not uploaded_file:
        st.info("👈 Carregue um CSV/XLSX na barra lateral.")
        st.stop()

    # Leitura e limpeza
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("São necessárias pelo menos duas colunas (tempo e resposta).")
        st.stop()

    c1, c2 = st.sidebar.columns(2)
    t_col = c1.selectbox("Coluna de tempo", cols, index=0)
    y_col = c2.selectbox("Coluna de resposta", cols, index=1)

    # limpeza robusta de vírgula/ponto
    t_clean = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."),
                            errors="coerce")
    y_clean = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."),
                            errors="coerce")

    mask = ~np.isnan(t_clean) & ~np.isnan(y_clean)
    t_data = t_clean[mask].values
    y_data = y_clean[mask].values

    # ordena por tempo
    idx_sort = np.argsort(t_data)
    t_data = t_data[idx_sort]
    y_data = y_data[idx_sort]

    if len(t_data) < 5:
        st.error("Poucos dados válidos após limpeza. Verifique o arquivo.")
        st.stop()

    st.sidebar.success(f"{len(t_data)} pontos válidos após processamento.")

    st.subheader("Dados experimentais")
    fig_raw, ax_raw = plt.subplots()
    ax_raw.scatter(t_data, y_data)
    ax_raw.set_xlabel("Tempo")
    ax_raw.set_ylabel("Resposta")
    ax_raw.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_raw)

    # -----------------------------
    # 5.2 Execução do ajuste
    # -----------------------------
    if not st.button("🚀 Rodar ajuste automático (Gompertz x Boltzmann)"):
        st.stop()

    status = st.empty()
    status.write("Ajustando modelo Gompertz...")
    gompertz_best = fit_model_family(
        t_data, y_data,
        model_func=gompertz_step,
        model_name="Gompertz",
        max_phases=max_phases,
        de_maxiter=de_maxiter,
        random_seed=random_seed
    )

    status.write("Ajustando modelo Boltzmann...")
    boltzmann_best = fit_model_family(
        t_data, y_data,
        model_func=boltzmann_step,
        model_name="Boltzmann",
        max_phases=max_phases,
        de_maxiter=de_maxiter,
        random_seed=random_seed + 1
    )

    if gompertz_best is None or boltzmann_best is None:
        st.error("Falha em pelo menos um dos ajustes. Verifique os dados e tente reduzir Nₘₐₓ.")
        st.stop()

    # -----------------------------
    # 5.3 Escolha do melhor modelo geral
    # -----------------------------
    # Usa o mesmo critério (AIC/AICc/BIC) que foi usado dentro de cada família
    candidates = [gompertz_best, boltzmann_best]
    overall_best = min(candidates, key=lambda d: d["ic_value"])

    status.write("Concluído ✅")

    # -----------------------------
    # 5.4 Resultados resumidos
    # -----------------------------
    st.markdown("## Resultado global")

    col_res_sum, col_res_tbl = st.columns([1, 2])

    with col_res_sum:
        st.markdown(f"""
**Melhor modelo:** `{overall_best['model_name']}`  
**Nº de fases:** `{overall_best['n_phases']}`  
**Critério usado:** `{overall_best['ic_used']}`  
**Valor do critério:** `{overall_best['ic_value']:.3f}`  
**R²:** `{overall_best['r2']:.4f}`  
**Loss (Lorentz):** `{overall_best['loss']:.4f}`  
**SSE:** `{overall_best['sse']:.4f}`
""")

    # Tabela de parâmetros do melhor modelo
    theta = overall_best["theta_hat"]
    se = overall_best["theta_se"]
    n_phases_best = overall_best["n_phases"]

    rows = []
    for i in range(n_phases_best):
        idx = i * 3
        A, mu, lam = theta[idx:idx + 3]
        seA, semu, selam = se[idx:idx + 3] if se[idx:idx+3].size == 3 else (np.nan, np.nan, np.nan)
        rows.append({
            "Fase": i + 1,
            "A (assíntota)": f"{A:.4f}",
            "SE(A)": f"{seA:.4f}" if np.isfinite(seA) else "-",
            "μ (taxa máx)": f"{mu:.4f}",
            "SE(μ)": f"{semu:.4f}" if np.isfinite(semu) else "-",
            "λ (lag)": f"{lam:.4f}",
            "SE(λ)": f"{selam:.4f}" if np.isfinite(selam) else "-"
        })

    with col_res_tbl:
        st.subheader("Parâmetros do modelo vencedor")
        st.table(pd.DataFrame(rows))

    # -----------------------------
    # 5.5 Gráfico do modelo vencedor
    # -----------------------------
    st.subheader("Ajuste do modelo vencedor")

    t_smooth = np.linspace(t_data.min(), t_data.max(), 400)
    y_smooth_total = polyauxic_model(
        t_smooth,
        overall_best["theta_hat"],
        overall_best["model_func"],
        overall_best["n_phases"]
    )

    fig_best, ax_best = plt.subplots(figsize=(8, 5))
    ax_best.scatter(t_data, y_data, color="black", alpha=0.6, label="Experimental", s=30)
    ax_best.plot(t_smooth, y_smooth_total, linestyle="-", linewidth=2,
                 label=f"{overall_best['model_name']} (total)")

    # fases individuais
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for i in range(overall_best["n_phases"]):
        idx = i * 3
        A, mu, lam = overall_best["theta_hat"][idx:idx + 3]
        y_phase = overall_best["model_func"](t_smooth, A, mu, lam)
        ax_best.plot(
            t_smooth, y_phase,
            linestyle="--",
            linewidth=1.5,
            color=colors[i % len(colors)],
            label=f"Fase {i + 1}"
        )

    ax_best.set_xlabel("Tempo")
    ax_best.set_ylabel("Resposta")
    ax_best.grid(True, linestyle=":", alpha=0.6)
    ax_best.legend()
    st.pyplot(fig_best)

    # -----------------------------
    # 5.6 Detalhe dos dois modelos para comparação
    # -----------------------------
    st.markdown("## Comparação entre Gompertz e Boltzmann (melhor de cada família)")

    def summary_dict(res):
        return {
            "Modelo": res["model_name"],
            "Fases": res["n_phases"],
            "Critério usado": res["ic_used"],
            "Valor critério": f"{res['ic_value']:.3f}",
            "R²": f"{res['r2']:.4f}",
            "AIC": f"{res['AIC']:.3f}",
            "AICc": f"{res['AICc']:.3f}",
            "BIC": f"{res['BIC']:.3f}",
            "Loss (Lorentz)": f"{res['loss']:.4f}",
            "SSE": f"{res['sse']:.4f}"
        }

    df_comp = pd.DataFrame([summary_dict(gompertz_best), summary_dict(boltzmann_best)])
    st.table(df_comp)


if __name__ == "__main__":
    main()
