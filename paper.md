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

The **Polyauxic Modeling Platform** is an open-source Python tool designed to automate the kinetic modeling of complex microbial growth patterns, specifically focusing on mono- and polyauxic (multiphasic) behaviors. In predictive microbiology and industrial biotechnology, accurately estimating kinetic parameters (such as the maximum specific growth rate ($r_{max}$) and the lag phase duration ($\lambda$)) is critical for process control and scale-up.

While classical empirical models effectively describe simple single-phase growth, they fail to capture the sequential substrate consumption (diauxie, triauxie) often observed in complex media, such as lignocellulosic hydrolysates or industrial wastewaters. This software implements the semi-mechanistic framework proposed by @Mockaitis:2025, treating polyauxic growth as a weighted summation of reformulated sigmoidal phases. It features a high-performance hybrid optimization engine to resolve the multi-modal parameter landscapes inherent to these complex models, accessible via a user-friendly Streamlit interface.

# Statement of Need

Modeling multiphasic microbial growth presents significant computational challenges that are not adequately addressed by standard statistical software or general-purpose fitting libraries:

1. **High-Dimensional Non-Convexity:** Polyauxic models considered in this code shows $n$ phases, containing $3n + 2$ parameters. The objective function typically exhibits multiple local minima, making standard gradient-descent algorithms highly sensitive to initial guesses.
2. **Constraint Handling:** Biologically meaningful solutions require strict adherence to physical and physiological boundaries. Implementing these dynamic constraints in general-purpose tools is non-trivial for experimentalists, yet essential to avoid mathematical artifacts. The platform enforces:
    * **Non-negativity:** All kinetic parameters (rates, constants, and yields) must satisfy $\theta_i \ge 0$.
    * **Strict Temporal Ordering:** To represent sequential substrate consumption, the lag time of a subsequent phase must strictly follow the previous one ($\lambda_1 < \lambda_2 < \dots < \lambda_n$). This prevents the logical fallacy of a secondary metabolic phase starting before the primary phase is established.
    * **Softmax Constraints:** The algorithm employs softmax transformations to enforce parameter constraints, such as normalization or competitive allocation. By mapping optimization variables to a probability simplex (where $\sum p_i = 1$), this technique converts rigid boundary conditions into differentiable functions, ensuring numerical stability without the need for hard clipping.
3. **Reproducibility:** Manual heuristic initialization ("eyeballing" start parameters) introduces bias. A robust tool must offer an automated, deterministic workflow to ensure that results are reproducible across different research groups.
4. **Outlier Removal:** Biological datasets are inherently noisy due to factors such as sensor drift, bubbles in bioreactors, or off-line sampling errors. Standard least-squares regression is highly sensitive to these anomalies, where a single outlier can disproportionately skew the fitted curve. The platform integrates an automated outlier detection step to distinguish between genuine metabolic shifts and measurement artifacts, ensuring that the estimated kinetic parameters ($\mu_{max}$, $K_s$) reflect the true physiology of the organism rather than experimental noise.

The **Polyauxic Modeling Platform** addresses these needs by providing a Python-based workflow that integrates semi-mechanistic model formulations, automated outlier detection using the Robust Regression and Outlier Removal (ROUT) method [@Motulsky:2006], and a hybrid optimization strategy that eliminates the need for manual parameter initialization.

# Software Design and Implementation

The software is architected as a modular Python library that decouples the mathematical core from the user interface, ensuring testability and extensibility. The computational pipeline consists of three distinct stages: data preprocessing, hybrid optimization, and statistical model selection.

## 1. Semi-Mechanistic Mathematical Models

The core library implements reformulated versions of sigmoid functions where the parameters of biological interest ($r_{max}$ and $\lambda$) appear explicitly, adapting the reparameterization logic proposed by Zwietering et al. [@Zwietering:1990]. For a polyauxic system with $n$ phases, the cumulative biomass or product formation $y(x)$ is modeled as a weighted summation of individual growth phases.

**Modified Polyauxic Boltzmann Model:**

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \left[ \frac{1}{1 + \exp\left(\frac{4 \cdot r_{max,j} \cdot (\lambda_j - x)}{(y_f - y_i) \cdot p_j} + 2\right)} \right]$$

**Modified Polyauxic Gompertz Model:**

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \cdot \exp\left( - \exp\left( \frac{r_{max,j} \cdot e}{(y_f - y_i) \cdot p_j} (\lambda_j - x) + 1 \right) \right)$$

Where $y_i$ and $y_f$ represent the initial and final asymptotic values, respectively. The term $p_j$ denotes the fractional contribution of phase $j$ to the total growth. The scaling of the denominator by $p_j$ ensures that the estimated parameter $r_{max,j}$ reflects the true maximum rate for that specific phase, rather than a global artifact.

## 2. Robust Optimization Engine

Solving non-linear regression problems for polyauxic models is computationally challenging due to the high dimensionality of the parameter space ($3n+2$ parameters) and the presence of multiple local minima. The software employs a **Hybrid Optimization Strategy** using `scipy.optimize` [@Virtanen:2020]:

1.  **Stochastic Global Search (Differential Evolution):** The pipeline begins with Differential Evolution (DE) [@Storn:1997], a genetic algorithm that explores the global parameter space. DE is robust against local minima and does not require gradient information, making it ideal for finding the basin of attraction for the optimal solution without user-supplied initial guesses.
2.  **Gradient-Based Local Refinement:** The best candidate vector from the DE stage is passed as the initialization point to the **L-BFGS-B** algorithm [@Byrd:1995]. This quasi-Newton method utilizes gradient information to polish the parameters to high precision, ensuring convergence to the exact local minimum.
3.  **Constraint Handling via Softmax:** To strictly enforce the physical constraint that the sum of phase contributions must equal unity ($\sum p_j = 1$), the software utilizes a Softmax transformation internally. This maps the unbounded optimization variables to a valid probability simplex, ensuring stability during the fitting process.

## 3. Statistical Analysis and Model Selection

The software implements a rigorous statistical framework to validate the fit and prevent overfitting:

* **Automated Outlier Detection:** Prior to fitting, the module executes a Z-score based filter to identify and flag data points that deviate significantly from the local trend, mitigating the impact of experimental noise (e.g., sensor drift or sampling errors).
* **Information Criteria:** Model parsimony is evaluated using both the Akaike Information Criterion (AIC/AICc) [@Akaike:1974; @Hurvich:1989] and the Bayesian Information Criterion (BIC) [@Schwarz:1978]. While AIC is generally preferred for predictive accuracy, BIC introduces a stronger penalty for the number of parameters, providing a more conservative selection of phases in larger datasets. The software calculates both, allowing the user to select the optimal number of phases ($n$) based on the specific context of their experimental design.

# Research Impact Statement

The **Polyauxic Modeling Platform** establishes a rigorous computational standard for analyzing multi-substrate microbial growth, addressing a significant gap in the bioprocess engineering toolkit. While commercial software and ad-hoc scripts exist for simple Monod kinetics, the analysis of complex polyauxic behaviors—ubiquitous in industrial fermentations using complex media—has historically relied on manual data segmentation or heuristic "eyeballing." This package transforms these subjective workflows into an objective, reproducible, and automated pipeline.

Originally developed to support specific research in bioenergy and kinetic modeling, the software has expanded to serve the broader metabolic engineering community. It provides critical infrastructure for researchers working with non-conventional yeasts and bacteria where complex regulatory mechanisms (such as catabolite repression) render standard models insufficient. By building directly upon the widely adopted Scientific Python ecosystem (`scipy`, `pandas`, `matplotlib`), the platform ensures seamless integration into modern data science workflows, allowing users to move from raw bioreactor data to publication-quality kinetic parameters without friction.

Beyond its utility in research, the platform is designed to lower the barrier to entry for advanced mathematical modeling. The decoupling of the core library from the web-based visualization layer (Streamlit) democratizes access to high-dimensional non-linear regression techniques. This dual architecture allows experimentalists with limited programming experience to validate kinetic hypotheses robustly, while simultaneously offering computational biologists a modular library for high-throughput analysis. By standardizing the handling of outliers and parameter constraints, the **Polyauxic Modeling Platform** serves as a foundational tool for ensuring reproducibility in fermentation science.

# Acknowledgements

This work was supported by the São Paulo Research Foundation (FAPESP), grant number 2018/18802-0, and by the National Council for Scientific and Technological Development (CNPq), grant numbers 425241/2018-1 and 409407/2023-2.

# References
