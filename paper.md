---
title: 'Polyauxic Modeling Platform: A Python Tool for Semi-Mechanistic Microbial Growth Kinetics'
tags:
  - Python
  - microbiology
  - kinetic modeling
  - nonlinear regression
  - bioprocess optimization
  - streamlit
  - scipy
authors:
  - name: Gustavo Mockaitis
    orcid: 0000-0002-4231-1056
    affiliation: 1
affiliations:
 - name: Interdisciplinary Research Group on Biotechnology Applied to the Agriculture and the Environment (GBMA), School of Agricultural Engineering, University of Campinas (UNICAMP), Brazil
   index: 1
date: 16 January 2026
bibliography: paper.bib
---

# Summary

The **Polyauxic Modeling Platform** is an open-source Python tool designed to automate the kinetic modeling of complex microbial growth patterns, specifically focusing on mono- and polyauxic (multiphasic) behaviors. In predictive microbiology and industrial biotechnology, accurately estimating kinetic parameters—such as the maximum specific growth rate ($r_{max}$) and the lag phase duration ($\lambda$)—is critical for process control and scale-up.

While classical empirical models effectively describe simple single-phase growth, they fail to capture the sequential substrate consumption (diauxie, triauxie) often observed in complex media, such as lignocellulosic hydrolysates or industrial wastewaters. This software implements the semi-mechanistic framework proposed by @Mockaitis:2025, treating polyauxic growth as a weighted summation of reformulated sigmoidal phases. It features a high-performance hybrid optimization engine to resolve the multi-modal parameter landscapes inherent to these complex models, accessible via a user-friendly Streamlit interface.

# Statement of Need

Modeling multiphasic microbial growth presents significant computational challenges that are not adequately addressed by standard statistical software or general-purpose fitting libraries:

1.  **High-Dimensional Non-Convexity:** Polyauxic models with $n$ phases often contain $3n + 2$ parameters. The objective function typically exhibits multiple local minima, making standard gradient-descent algorithms highly sensitive to initial guesses.
2.  **Constraint Handling:** Biologically meaningful solutions require strict temporal ordering of phases (e.g., $\lambda_1 < \lambda_2 < \dots < \lambda_n$). Implementing these dynamic inequality constraints in general-purpose tools is non-trivial for experimentalists.
3.  **Reproducibility:** Manual heuristic initialization ("eyeballing" start parameters) introduces bias. A robust tool must offer an automated, deterministic workflow to ensure that results are reproducible across different research groups.

The **Polyauxic Modeling Platform** addresses these needs by providing a Python-based workflow that integrates semi-mechanistic model formulations, automated outlier detection using the Robust Regression and Outlier Removal (ROUT) method [@Motulsky:2006], and a hybrid optimization strategy that eliminates the need for manual parameter initialization.

# Mathematics and Implementation

## Semi-Mechanistic Formulations

The software implements reformulated versions of sigmoid functions where the parameters of interest ($r_{max}$ and $\lambda$) appear explicitly, based on the reparameterization logic for monoauxic growth [@Zwietering:1990]. For a polyauxic system with $n$ phases, the biomass concentration or product formation $y(x)$ is modeled as a weighted summation.

For the **Modified Polyauxic Boltzmann** model, the equation is implemented as:

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \left[ \frac{1}{1 + \exp\left(\frac{4 \cdot r_{max,j} \cdot (\lambda_j - x)}{(y_f - y_i) \cdot p_j} + 2\right)} \right]$$

For the **Modified Polyauxic Gompertz** model, the implementation follows:

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \cdot \exp\left( - \exp\left( \frac{r_{max,j} \cdot e}{(y_f - y_i) \cdot p_j} (\lambda_j - x) + 1 \right) \right)$$

Where $y_i$ and $y_f$ are initial and final asymptotic values, and $p_j$ is the fractional contribution of phase $j$ (subject to $\sum p_j = 1$). The scaling by $p_j$ in the denominator of the exponent ensures that $r_{max,j}$ represents the true maximum rate for that specific phase, rather than a global artifact.

## Hybrid Optimization Engine

To solve the non-linear regression problem without user-supplied guesses, the software employs a two-stage strategy using `scipy.optimize` [@Virtanen:2020]:

1.  **Global Search (Stochastic):** The algorithm first employs **Differential Evolution (DE)** [@Storn:1997], a stochastic genetic algorithm. DE is robust against local minima and explores the global parameter space to find the basin of attraction for the optimal solution.
2.  **Local Refinement (Gradient-based):** The best candidate vector from the DE stage is passed as the initial guess to the **L-BFGS-B** algorithm [@Byrd:1995]. This step polishes the parameters to high precision and ensures convergence to the exact local minimum.

Model parsimony is enforced via Akaike Information Criterion (AIC/AICc) [@Akaike:1974; @Hurvich:1989] to automatically select the optimal number of phases ($n$).

# Acknowledgements

The author gratefully acknowledges his mentors, especially Prof. Walter Borzani (*in memoriam*), Eugenio Foresti, Marcelo Zaiat, José Alberto Domingues Rodrigues, and Serge Roger Guiot, for their invaluable guidance to himself and generations of scientists in the study of microbial growth kinetics in complex systems.

This work was supported by the São Paulo Research Foundation (FAPESP), grant number 2018/18802-0, and by the National Council for Scientific and Technological Development (CNPq), grant numbers 425241/2018-1 and 409407/2023-2.

# References
