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

The **Polyauxic Modeling Platform** is an open-source Python tool designed to automate the kinetic modeling of complex microbial growth patterns, specifically focusing on mono- and polyauxic (multiphasic) behaviors. In predictive microbiology, accurately estimating kinetic parameters (such as the maximum specific growth rate - $r_{max}$ and lag phase duration - $\lambda$) is critical for process control.

While classical empirical models effectively describe simple single-phase growth, they fail to capture the sequential substrate consumption (diauxie, triauxie) often observed in complex media, such as lignocellulosic hydrolysates or industrial wastewaters. This software implements the semi-mechanistic framework proposed by @Mockaitis:2025, treating polyauxic growth as a weighted summation of reformulated sigmoidal phases. It features a high-performance hybrid optimization engine to resolve multi-modal parameter landscapes, accessible via a user-friendly Streamlit interface.

# Statement of Need

Modeling multiphasic microbial growth presents computational challenges often overlooked by general-purpose statistical software:

1. **High-Dimensional Non-Convexity:** Polyauxic models with $n$ phases involve $3n + 2$ parameters. The objective function typically exhibits multiple local minima, rendering standard gradient-descent algorithms highly sensitive to initial guesses.
2. **Constraint Handling:** Biologically meaningful solutions require strict adherence to physical boundaries. The platform enforces:
    * **Non-negativity:** All kinetic parameters must satisfy $\theta_i \ge 0$.
    * **Temporal Ordering:** Phase lags must be sequential ($\lambda_1 < \lambda_2 < \dots < \lambda_n$) to prevent logical fallacies in metabolic shifts.
    * **Softmax Constraints:** To ensure stability without hard clipping, the algorithm employs softmax transformations, mapping variables to a probability simplex (where $\sum p_i = 1$).
3. **Reproducibility:** Manual heuristic initialization ("eyeballing") introduces bias. A robust tool must offer an automated, deterministic workflow to ensure result consistency across research groups.
4. **Outlier Removal:** Biological datasets are inherently noisy. Standard least-squares regression is sensitive to artifacts (e.g., sensor drift). The platform integrates automated outlier detection to distinguish metabolic shifts from experimental noise.

The **Polyauxic Modeling Platform** addresses these needs by integrating semi-mechanistic formulations, automated outlier detection via the ROUT method [@Motulsky:2006], and a hybrid optimization strategy that eliminates manual initialization.

# Software Design and Implementation

The software is architected as a modular Python library decoupling the mathematical core from the visualization layer, ensuring testability. The pipeline consists of three stages:

## 1. Semi-Mechanistic Mathematical Models

The library implements reformulated sigmoid functions where biological parameters ($r_{max}$, $\lambda$) appear explicitly [@Zwietering:1990]. For a polyauxic system with $n$ phases, biomass $y(x)$ is modeled as a weighted summation:

**Modified Polyauxic Boltzmann Model:**

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \left[ \frac{1}{1 + \exp\left(\frac{4 \cdot r_{max,j} \cdot (\lambda_j - x)}{(y_f - y_i) \cdot p_j} + 2\right)} \right]$$

**Modified Polyauxic Gompertz Model:**

$$y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \cdot \exp\left( - \exp\left( \frac{r_{max,j} \cdot e}{(y_f - y_i) \cdot p_j} (\lambda_j - x) + 1 \right) \right]$$

Here, $y_i$ and $y_f$ are asymptotic values, and $p_j$ denotes the fractional contribution of phase $j$. The scaling by $p_j$ ensures $r_{max,j}$ reflects the true maximum rate for that specific phase.

## 2. Robust Optimization Engine

Solving these non-linear problems requires navigating a complex landscape. The software employs a **Hybrid Optimization Strategy** using `scipy.optimize` [@Virtanen:2020]:

1.  **Global Search:** Differential Evolution (DE) [@Storn:1997] explores the parameter space to locate the basin of attraction, avoiding local minima without requiring initial guesses.
2.  **Local Refinement:** The best DE candidate initializes the **L-BFGS-B** algorithm [@Byrd:1995], using gradient information to polish parameters to high precision.
3.  **Constraint Enforcement:** Internal Softmax transformations map unbounded variables to valid probability distributions ($\sum p_j = 1$), ensuring stoichiometric consistency during fitting.

## 3. Statistical Analysis

To prevent overfitting, the software implements a rigorous framework:
* **Outlier Detection:** A Z-score based filter flags data points deviating significantly from local trends prior to fitting.
* **Model Selection:** Parsimony is evaluated using Akaike (AIC/AICc) [@Akaike:1974] and Bayesian Information Criteria (BIC) [@Schwarz:1978]. While AIC prioritizes predictive accuracy, BIC imposes stronger penalties for parameter count, aiding the selection of the optimal phase number ($n$).

# Research Impact Statement

The **Polyauxic Modeling Platform** establishes a computational standard for analyzing multi-substrate microbial growth, filling a gap in the bioprocess engineering toolkit. Analysis of complex polyauxic behaviors has historically relied on manual segmentation. This package transforms these subjective workflows into an objective, reproducible pipeline.

Originally developed for bioenergy research, the software now serves the broader metabolic engineering community, particularly for non-conventional organisms with complex regulation. By building on the Scientific Python ecosystem (`scipy`, `pandas`), it integrates seamlessly into modern data workflows.

The platform lowers the barrier to advanced modeling. Decoupling the core library from the Streamlit visualization layer allows experimentalists to validate hypotheses via the web interface, while computational biologists can utilize the modular library for high-throughput analysis. By standardizing outlier handling and constraints, the platform serves as a foundational tool for reproducibility in fermentation science.

# AI Usage Statement

During the preparation of this work, Artificial Intelligence tools based on Large Language Models (LLMs) were used exclusively to assist with text refinement, structural organization of the manuscript, and code debugging. All scientific concepts, experimental designs, data analysis, and final validation of the results are the sole responsibility of the human authors.

# Acknowledgements

This work was supported by the São Paulo Research Foundation (FAPESP), grant number 2018/18802-0, and by the National Council for Scientific and Technological Development (CNPq), grant numbers 425241/2018-1 and 409407/2023-2.

# References
