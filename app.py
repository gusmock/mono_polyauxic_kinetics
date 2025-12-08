"""
                                                                                               @@@@                      
                    ::++        ++..                                       ######  ########  @@@@@@@@                   
                    ++++      ..++++                                     ##########  ########  @@@@                    
                    ++++++    ++++++                                 #####  ########  ##########  ####                  
          ++        ++++++++++++++++      ++++                    ########  ########  ########   ########                
        ++++++mm::++++++++++++++++++++  ++++++--                ##########  ########  ########  ##########              
          ++++++++++mm::########::++++++++++++                ##  ##########  ######  ######   ##########  ##            
            ++++++::####        ####++++++++                 #####  ########  ######  ######  ########  #######            
          --++++MM##      ####      ##::++++                ########  ########  ####  ####   ########  ##########          
    ++--  ++++::##    ##    ##  ..MM  ##++++++  ::++       ###########  ######  ####  ####  ######  ##############         
  --++++++++++##    ##          @@::  mm##++++++++++          ###########  ###### ##  ####  ####  ##############        
    ++++++++::##    ##          ##      ##++++++++++      ###   ###########  ####  ##  ##  ####  ############    ##        
        ++++@@++              --        ##++++++          ######    ########  ##          ##  ########    #########      
        ++++##..      MM  ..######--    ##::++++          ##########      ####              ######    #############      
        ++++@@++    ####  ##########    ##++++++          ################                  ######################      
    ++++++++::##          ##########    ##++++++++++      ##################                  #################  @@@@@  
  ::++++++++++##    ##      ######    mm##++++++++++                                                            @@@@@@@
    mm++::++++++##  ##++              ##++++++++++mm        ################                  #################  @@@@@  
          ++++++####                ##::++++                ##############                    ##################        
            ++++++MM##@@        ####::++++++                 #######    ######              ##################          
          ++++++++++++@@########++++++++++++mm                #     ########  ##          ##  ##############            
        mm++++++++++++++++++++++++++++--++++++                  ##########  ############  ####  ########                
          ++::      ++++++++++++++++      ++++                    ######  ######################  ####                  
                    ++++++    ++++++                                    ##################    ####                      
                    ++++      ::++++                                    ##############  @@@@@                         
                    ++++        ++++                                                   @@@@@@@                          
                                                                                        @@@@@ 
================================================================================
POLYAUXIC ROBUSTNESS SIMULATOR (v3.2 - ROUT & Visualization)
================================================================================

Author: Prof. Dr. Gustavo Mockaitis (GBMA/FEAGRI/UNICAMP)
GitHub: https://github.com/gusmock/mono_polyauxic_kinetics/

DESCRIPTION:
This Streamlit application performs rigorous Monte Carlo robustness testing for 
Polyauxic Kinetic Models. It incorporates the ROUT method (Robust regression 
and Outlier removal) based on Motulsky & Brown (2006) to handle outliers 
scientifically using False Discovery Rate (FDR).

REFERENCES:
1. Motulsky, H. J., & Brown, R. E. (2006). Detecting outliers when fitting 
   data with nonlinear regression – a new method based on robust nonlinear 
   regression and the false discovery rate. BMC Bioinformatics, 7, 123.
2. Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute 
   Deviation. Journal of the American Statistical Association.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# 0. ROUT-like Outlier Detection (MAD-based)
# ------------------------------------------------------------
def detect_outliers(y_true, y_pred):
    """
    Detecta outliers baseando-se no Desvio Absoluto da Mediana (MAD).
    Utiliza um Z-score modificado com corte em 2.5 (aproximação ROUT-like simplificada).
    """
    residuals = y_true - y_pred
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    # Constante 1.4826 torna o MAD consistente com o desvio padrão para distribuição normal
    sigma = 1.4826 * mad if mad > 1e-12 else 1e-12
    z = np.abs(residuals - med) / sigma
    return z > 2.5

# ------------------------------------------------------------
# 1. Model Equations
# ------------------------------------------------------------
def boltzmann_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    # Evitar divisão por zero e overflow
    expo = (4 * r_j * (lam_j - t)) / (dy * p) + 2
    expo = np.clip(expo, -500, 500) 
    return p / (1 + np.exp(expo))

def gompertz_term(t, y_i, y_f, p_j, r_j, lam_j):
    t = np.asarray(t)
    dy = y_f - y_i if abs(y_f - y_i) > 1e-12 else 1e-12
    p = max(p_j, 1e-12)
    expo = (r_j * np.e * (lam_j - t)) / (dy * p) + 1
    expo = np.clip(expo, -500, 500)
    return p * np.exp(-np.exp(expo))

# ------------------------------------------------------------
# 2. Polyauxic generating function (Updated for separate terms)
# ------------------------------------------------------------
def polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    """Retorna a soma e os termos individuais para plotagem."""
    p_vec = np.asarray(p_vec, dtype=float)
    # Normalização forçada
    if np.sum(p_vec) <= 0:
        p_vec = np.ones_like(p_vec) / len(p_vec)
    else:
        p_vec = p_vec / np.sum(p_vec)
    
    components = []
    sum_terms = np.zeros_like(t, dtype=float)
    
    for j in range(len(p_vec)):
        term = func(t, y_i, y_f, p_vec[j], r_vec[j], lam_vec[j])
        components.append((y_f - y_i) * term) # Componente escalado
        sum_terms += term
        
    y_total = y_i + (y_f - y_i) * sum_terms
    return y_total, components, p_vec

def polyauxic_generate(t, y_i, y_f, p_vec, r_vec, lam_vec, func):
    # Wrapper simples para compatibilidade com o código antigo de Monte Carlo
    y, _, _ = polyauxic_components(t, y_i, y_f, p_vec, r_vec, lam_vec, func)
    return y

# ------------------------------------------------------------
# 3. Fitting & Helpers (Mantidos iguais)
# ------------------------------------------------------------
def polyauxic_fit_model(t, theta, func, n_phases):
    y_i = theta[0]
    y_f = theta[1]
    z = theta[2 : 2 + n_phases]
    r = theta[2 + n_phases : 2 + 2*n_phases]
    lam = theta[2 + 2*n_phases : 2 + 3*n_phases]
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    p = expz / np.sum(expz)
    sum_terms = np.zeros_like(t)
    for j in range(n_phases):
        sum_terms += func(t, y_i, y_f, p[j], r[j], lam[j])
    return y_i + (y_f - y_i) * sum_terms

def sse_loss(theta, t, y, func, n_phases):
    y_pred = polyauxic_fit_model(t, theta, func, n_phases)
    # Penalidade suave se predição negativa extrema
    if np.any(y_pred < -0.1*np.max(np.abs(y))): return 1e12
    return np.sum((y - y_pred)**2)

def smart_guess(t, y, n_phases):
    dy = np.gradient(y, t)
    dy_s = np.convolve(dy, np.ones(5)/5, mode='same') if len(dy)>=5 else dy
    peaks, props = find_peaks(dy_s, height=np.max(dy_s)*0.1 if np.max(dy_s)>0 else 0)
    guesses = []
    if len(peaks) > 0:
        idx = np.argsort(props['peak_heights'])[::-1][:n_phases]
        best = peaks[idx]
        for p in best: guesses.append((t[p], abs(dy_s[p])))
    while len(guesses) < n_phases:
        tspan = t.max() - t.min() if t.max() > t.min() else 1
        guesses.append((t.min() + tspan*(len(guesses)+1)/(n_phases+1), (y.max()-y.min())/(tspan/n_phases)))
    guesses = sorted(guesses, key=lambda x: x[0])
    theta0 = np.zeros(2 + 3*n_phases)
    theta0[0] = y.min(); theta0[1] = y.max()
    for i,(lam,r) in enumerate(guesses):
        theta0[2 + n_phases + i] = r
        theta0[2 + 2*n_phases + i] = lam
    return theta0

def fit_polyauxic(t_all, y_all, func, n_phases):
    t_scale = np.max(t_all) if np.max(t_all)>0 else 1
    y_scale = np.max(np.abs(y_all)) if np.max(np.abs(y_all))>0 else 1
    t_n = t_all / t_scale
    y_n = y_all / y_scale
    theta0 = smart_guess(t_all, y_all, n_phases)
    th0 = np.zeros_like(theta0)
    th0[0] = theta0[0]/y_scale; th0[1] = theta0[1]/y_scale
    th0[2:2+n_phases] = 0 
    th0[2+n_phases:2+2*n_phases] = theta0[2+n_phases:2+2*n_phases]*(t_scale/y_scale)
    th0[2+2*n_phases:2+3*n_phases] = theta0[2+2*n_phases:2+3*n_phases]/t_scale
    
    bounds = [(-0.2,1.5),(0,2)] + [(-10,10)]*n_phases + [(0,500)]*n_phases + [(-0.1,1.2)]*n_phases
    popsize=20
    init_pop = np.tile(th0,(popsize,1))*(np.random.uniform(0.8,1.2,(popsize,len(th0))))
    
    res_de = differential_evolution(sse_loss, bounds, args=(t_n,y_n,func,n_phases),
                                    maxiter=800, popsize=popsize, init=init_pop, strategy="best1bin", polish=True, tol=1e-6)
    res = minimize(sse_loss, res_de.x, args=(t_n,y_n,func,n_phases), method="L-BFGS-B", bounds=bounds, tol=1e-10)
    
    th_n = res.x
    th = np.zeros_like(th_n)
    th[0]=th_n[0]*y_scale; th[1]=th_n[1]*y_scale
    th[2:2+n_phases]=th_n
# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")

profile_pic_url = "https://github.com/gusmock.png" 

footer_html = f"""
<style>
    .footer-container {{
        width: 100%;
        font-family: sans-serif;
        color: #444;
        margin-bottom: 20px;
    }}
    
    /* Layout Flex para Foto + Texto */
    .profile-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 15px;
    }}
    
    /* Estilo da Foto */
    .profile-img {{
        width: 80px;
        height: 80px;
        border-radius: 50%;       /* Deixa redonda */
        object-fit: cover;
        border: 2px solid #e0e0e0; /* Borda sutil */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .profile-info h4 {{
        margin: 0;
        font-size: 1.1rem;
        color: #222;
    }}
    
    .profile-info p {{
        margin: 2px 0 0 0;
        font-size: 0.9rem;
        color: #666;
    }}

    .badge-container {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
    }}
    
    .badge-container img {{
        height: 28px;
    }}
</style>

<div class="footer-container">
    
    <div class="profile-header">
        <img src="{profile_pic_url}" class="profile-img" alt="Gustavo Mockaitis">
        <div class="profile-info">
            <h2>Development: Prof. Dr. Gustavo Mockaitis</h2>
            <h4>GBMA / FEAGRi / UNICAMP</h4>
            <p>Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and Environment, School of Agricultural Engineering, University of Campinas (GBMA/FEAGRI/UNICAMP), 397 Michel Debrun Street, CEP 13.083-875, Campinas, SP, Brazil.</p>
        </div>
    </div>

    <div class="badge-container">
        <a href="https://arxiv.org/abs/2507.05960" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2507.05960-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
        </a>
        <a href="https://github.com/gusmock/mono_polyauxic_kinetics/" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://orcid.org/0000-0002-4231-1056" target="_blank">
            <img src="https://img.shields.io/badge/ORCID-iD-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID">
        </a>
        <a href="https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=pt-BR&oi=ao" target="_blank">
            <img src="https://img.shields.io/badge/Scholar-Perfil-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar">
        </a>
        <a href="https://www.researchgate.net/profile/Gustavo-Mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/ResearchGate-Perfil-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate">
        </a>
        <a href="http://lattes.cnpq.br/1400402042483439" target="_blank">
            <img src="https://img.shields.io/badge/Lattes-CV-003399?style=for-the-badge&logo=brasil&logoColor=white" alt="Lattes">
        </a>
        <a href="https://www.linkedin.com/in/gustavo-mockaitis/" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-Conectar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://www.webofscience.com/wos/author/record/J-7107-2019" target="_blank">
            <img src="https://img.shields.io/badge/Web_of_Science-Perfil-5E33BF?style=for-the-badge&logo=clarivate&logoColor=white" alt="Web of Science">
        </a>
        <a href="http://feagri.unicamp.br/mockaitis" target="_blank">
            <img src="https://img.shields.io/badge/UNICAMP-Institucional-CC0000?style=for-the-badge&logo=google-academic&logoColor=white" alt="UNICAMP">
        </a>
    </div>
</div>
"""

st_components.html(footer_html, height=280, scrolling=False)
