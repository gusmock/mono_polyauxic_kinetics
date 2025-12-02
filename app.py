import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ==============================================================================
# 1. MODELOS SIGMOIDAIS POLIAUXICOS (Eqs. 31 e 32 – METODOLOGIA OFICIAL)
# ==============================================================================

def boltzmann_phase_eq31(t, yi, yf, p, rmax, lam):
    """
    Termo j da Eq. 31:
        y(x) = y_i + (y_f - y_i) * Σ p_j / (1 + exp( 4 r_j^max (λ_j - x)/((y_f-y_i)p_j) + 2 ))

    Implementamos apenas o termo:
        termo_j = p_j / ( 1 + exp( 4 r_j^max (λ_j - x)/((y_f-y_i)p_j) + 2 ) )
    """
    delta_y = max(yf - yi, 1e-8)
    p_safe = max(p, 1e-12)

    exponent = 4.0 * rmax * (lam - t) / (delta_y * p_safe) + 2.0
    exponent = np.clip(exponent, -100.0, 100.0)

    return p_safe / (1.0 + np.exp(exponent))


def gompertz_phase_eq32(t, yi, yf, p, rmax, lam):
    """
    Termo j da Eq. 32:
        y(x) = y_i + (y_f - y_i) * Σ p_j exp( -exp( 1 + r_j^max e (λ_j - x)/((y_f-y_i)p_j) ) )

    Implementamos apenas o termo:
        termo_j = p_j * exp( -exp( 1 + r_j^max e (λ_j - x)/((y_f-y_i)p_j) ) )
    """
    delta_y = max(yf - yi, 1e-8)
    p_safe = max(p, 1e-12)

    exponent = 1.0 + (rmax * np.e) * (lam - t) / (delta_y * p_safe)
    exponent = np.clip(exponent, -100.0, 100.0)

    inner = np.exp(exponent)
    return p_safe * np.exp(-inner)


def polyauxic_model(t, theta, phase_func, n_phases):
    """
    Modelo completo das Eqs. 31 e 32.

    Parâmetros:
        θ = [ y_i, y_f,
              z_1..z_n,
              r_1^max..r_n^max,
              λ_1..λ_n ]

    p_j = softmax(z_j)
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)

    yi = theta[0]
    yf = theta[1]

    n = n_phases

    z = theta[2:2 + n]
    rmax = theta[2 + n:2 + 2*n]
    lam = theta[2 + 2*n:2 + 3*n]

    # softmax dos p_j
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    p = exp_z / np.sum(exp_z)

    # soma das fases
    y_red = 0.0
    for j in range(n):
        y_red += phase_func(t, yi, yf, p[j], rmax[j], lam[j])

    return yi + (yf - yi) * y_red


# ==============================================================================
# 2. FUNÇÕES DE PERDA E ESTATÍSTICAS
# ==============================================================================

def lorentzian_loss(params, t_exp, y_exp, model_func, n_phases):
    y_pred = polyauxic_model(t_exp, params, model_func, n_phases)
    residuals = y_exp - y_pred

    mad = np.median(np.abs(residuals - np.median(residuals)))
    scale = 1.4826 * mad if mad > 1e-12 else 1.0

    return np.sum(np.log(1.0 + (residuals / scale)**2))


def calculate_sse_r2(y_true, y_pred):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - sse/ss_tot if ss_tot > 0 else 0.0
    return sse, r2


def information_criteria(y_true, y_pred, n_params):
    n_data = len(y_true)
    sse, _ = calculate_sse_r2(y_true, y_pred)

    if sse <= 0:
        return np.inf, np.inf, np.inf, sse

    logL = -n_data/2 * (np.log(2*np.pi*sse/n_data) + 1)
    AIC = 2*n_params - 2*logL

    if n_data - n_params - 1 > 0:
        AICc = AIC + (2*n_params*(n_params+1)) / (n_data - n_params - 1)
    else:
        AICc = np.inf

    BIC = n_params * np.log(n_data) - 2*logL

    return AIC, AICc, BIC, sse


def select_ic_value(n_data, n_params, AIC, AICc, BIC):
    if n_data > 200:
        return BIC, "BIC"
    if n_data / n_params < 40:
        return AICc, "AICc"
    return AIC, "AIC"


# ==============================================================================
# 3. HESSIANA E ERROS-PADRÃO
# ==============================================================================

def numerical_hessian(func, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n,n))

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

            H[i,j] = (fpp - fpm - fmp + fmm)/(4*eps**2)

    return H


def parameter_uncertainty(theta_hat, t_data, y_data, model_func, n_phases):
    n_data = len(y_data)
    n_params = len(theta_hat)

    y_pred = polyauxic_model(t_data, theta_hat, model_func, n_phases)
    sse, _ = calculate_sse_r2(y_data, y_pred)

    if n_data <= n_params:
        return np.full_like(theta_hat, np.nan)

    sigma2 = sse / (n_data - n_params)

    def loss_fn(p):
        return lorentzian_loss(p, t_data, y_data, model_func, n_phases)

    try:
        H = numerical_hessian(loss_fn, theta_hat)
        C = sigma2 * np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(C)))
    except Exception:
        se = np.full_like(theta_hat, np.nan)

    return se


# ==============================================================================
# 4. AJUSTE AUTOMÁTICO (Gompertz Eq. 32 / Boltzmann Eq. 31)
# ==============================================================================

def fit_model_family(t_data, y_data, model_func, model_name, max_phases=4,
                     de_maxiter=500, random_seed=42):

    rng = np.random.default_rng(random_seed)
    n_data = len(y_data)
    y_min = np.min(y_data)
    y_max = np.max(y_data)
    t_max = np.max(t_data)

    best_result = None
    Cr = np.inf

    for n in range(1, max_phases+1):

        bounds = []

        # y_i
        bounds.append((0, y_max))
        # y_f
        bounds.append((0, 1.5*y_max))

        # z_j (softmax)
        for _ in range(n):
            bounds.append((-10, 10))

        # r_j^max
        for _ in range(n):
            bounds.append((0, (y_max/max(t_max,1e-6))*10))

        # lambda_j
        for _ in range(n):
            bounds.append((0, t_max))

        global_res = differential_evolution(
            lorentzian_loss,
            bounds,
            args=(t_data, y_data, model_func, n),
            maxiter=de_maxiter,
            popsize=15,
            tol=0.01,
            strategy='best1bin',
            seed=int(rng.integers(0,1_000_000)),
            polish=False
        )

        theta_global = global_res.x

        local_res = minimize(
            lorentzian_loss,
            theta_global,
            args=(t_data, y_data, model_func, n),
            method="Nelder-Mead",
            tol=1e-6
        )

        theta_hat = local_res.x
        y_pred = polyauxic_model(t_data, theta_hat, model_func, n)
        AIC, AICc, BIC, sse = information_criteria(y_data, y_pred, len(theta_hat))
        ic_value, ic_used = select_ic_value(n_data, len(theta_hat), AIC, AICc, BIC)
        _, r2 = calculate_sse_r2(y_data, y_pred)

        if ic_value < Cr:
            Cr = ic_value
            se = parameter_uncertainty(theta_hat, t_data, y_data, model_func, n)
            best_result = {
                "model_name": model_name,
                "model_func": model_func,
                "n_phases": n,
                "theta_hat": theta_hat,
                "theta_se": se,
                "loss": local_res.fun,
                "AIC": AIC,
                "AICc": AICc,
                "BIC": BIC,
                "ic_used": ic_used,
                "ic_value": ic_value,
                "sse": sse,
                "r2": r2
            }
        else:
            break

    return best_result


# ==============================================================================
# 5. INTERFACE STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Poliauxico – Eq. 31 e 32", layout="wide")
    st.title("Ajuste Poliauxico (Boltzmann Eq. 31 / Gompertz Eq. 32) – Metodologia Oficial")

    st.sidebar.header("Dados")
    uploaded_file = st.sidebar.file_uploader("Arquivo CSV/XLSX", type=["csv","xlsx"])

    max_phases = st.sidebar.number_input("Nº máximo de fases", 1,6,4)
    de_maxiter = st.sidebar.number_input("Maxiter DE", 50,2000,500,50)
    random_seed = st.sidebar.number_input("Seed", 0,999999,42)

    if not uploaded_file:
        st.info("Carregue um arquivo.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    except:
        st.error("Erro ao ler arquivo.")
        st.stop()

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("São necessárias duas colunas.")
        st.stop()

    t_col = st.sidebar.selectbox("Tempo", cols)
    y_col = st.sidebar.selectbox("Resposta", cols)

    t = pd.to_numeric(df[t_col].astype(str).str.replace(",", "."), errors="coerce")
    y = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors="coerce")

    mask = ~np.isnan(t) & ~np.isnan(y)
    t_data = np.asarray(t[mask])
    y_data = np.asarray(y[mask])

    idx = np.argsort(t_data)
    t_data = t_data[idx]
    y_data = y_data[idx]

    st.sidebar.success(f"{len(t_data)} pontos válidos")

    fig_raw, ax_raw = plt.subplots()
    ax_raw.scatter(t_data, y_data)
    ax_raw.set_xlabel("Tempo")
    ax_raw.set_ylabel("Resposta")
    st.pyplot(fig_raw)

    if not st.button("Rodar ajuste"):
        st.stop()

    st.write("Ajustando Gompertz (Eq. 32)...")
    gompertz_best = fit_model_family(
        t_data, y_data, gompertz_phase_eq32,
        "Gompertz (Eq. 32)", max_phases, de_maxiter, random_seed
    )

    st.write("Ajustando Boltzmann (Eq. 31)...")
    boltzmann_best = fit_model_family(
        t_data, y_data, boltzmann_phase_eq31,
        "Boltzmann (Eq. 31)", max_phases, de_maxiter, random_seed+1
    )

    if gompertz_best is None or boltzmann_best is None:
        st.error("Falha em algum ajuste.")
        st.stop()

    candidates = [gompertz_best, boltzmann_best]
    best = min(candidates, key=lambda d: d["ic_value"])

    st.subheader("Resultado Global")
    st.markdown(f"**Modelo:** {best['model_name']}")
    st.markdown(f"**Fases:** {best['n_phases']}")
    st.markdown(f"**Critério:** {best['ic_used']}")
    st.markdown(f"**Valor:** {best['ic_value']:.4f}")
    st.markdown(f"**R²:** {best['r2']:.4f}")
    st.markdown(f"**SSE:** {best['sse']:.4f}")
    st.markdown(f"**Loss:** {best['loss']:.4f}")

    theta = best["theta_hat"]
    se = best["theta_se"]
    n = best["n_phases"]

    yi = theta[0]
    yf = theta[1]
    z = theta[2:2+n]
    rmax = theta[2+n:2+2*n]
    lam = theta[2+2*n:2+3*n]

    # pesos p_j
    z_shift = z - np.max(z)
    p = np.exp(z_shift)/np.sum(np.exp(z_shift))

    # mostra parâmetros
    st.subheader("Parâmetros globais")
    st.write(f"y_i = {yi:.4f}")
    st.write(f"y_f = {yf:.4f}")

    st.subheader("Parâmetros das fases")
    rows = []
    for j in range(n):
        rows.append({
            "Fase": j+1,
            "p_j": f"{p[j]:.4f}",
            "r_j^max": f"{rmax[j]:.4f}",
            "λ_j": f"{lam[j]:.4f}"
        })

    st.table(pd.DataFrame(rows))

    st.subheader("Ajuste do modelo vencedor")
    t_smooth = np.linspace(t_data.min(), t_data.max(), 400)
    y_pred = polyauxic_model(t_smooth, theta, best["model_func"], n)

    fig_b, ax_b = plt.subplots()
    ax_b.scatter(t_data, y_data, color="black", s=30, label="Experimental")
    ax_b.plot(t_smooth, y_pred, label="Modelo total", linewidth=2)

    # fases individuais
    colors = ["tab:blue","tab:green","tab:orange","tab:purple","tab:brown","tab:red"]

    for j in range(n):
        y_red = best["model_func"](t_smooth, yi, yf, p[j], rmax[j], lam[j])
        y_phase = yi + (yf-yi)*y_red
        ax_b.plot(t_smooth, y_phase, "--", color=colors[j%6], label=f"Fase {j+1}")

    ax_b.legend()
    ax_b.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_b)

    st.subheader("Comparação Gompertz vs Boltzmann")
    def summarize(res):
        return {
            "Modelo": res["model_name"],
            "Fases": res["n_phases"],
            "Critério": res["ic_used"],
            "Valor": f"{res['ic_value']:.4f}",
            "R²": f"{res['r2']:.4f}",
            "AIC": f"{res['AIC']:.4f}",
            "AICc": f"{res['AICc']:.4f}",
            "BIC": f"{res['BIC']:.4f}",
            "Loss": f"{res['loss']:.4f}",
            "SSE": f"{res['sse']:.4f}"
        }

    st.table(pd.DataFrame([summarize(gompertz_best), summarize(boltzmann_best)]))


if __name__ == "__main__":
    main()
