# Polyauxic Modeling Platform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18025828.svg)](https://doi.org/10.5281/zenodo.18025828)
[![ArXiv](https://img.shields.io/badge/arXiv-2507.05960-b31b1b.svg)](https://arxiv.org/abs/2507.05960)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monopolyauxickinetics-test.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics**

This repository contains the source code for the **Polyauxic Modeling Platform**, a computational tool designed to fit semi-mechanistic kinetic models to complex microbial growth data. It implements the framework described in the preprint paper *Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics* (Mockaitis, 2025), capable of resolving multi-phasic (polyauxic) behavior in datasets common to environmental biotechnology and industrial bioprocesses.

## 📖 Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Theoretical Framework](https://www.google.com/search?q=%23theoretical-framework)
* [Features](https://www.google.com/search?q=%23features)
* [Installation](https://www.google.com/search?q=%23installation)
* [Usage](https://www.google.com/search?q=%23usage)
* [Data Format](https://www.google.com/search?q=%23data-format)
* [Citation](https://www.google.com/search?q=%23citation)
* [License](https://www.google.com/search?q=%23license)

## 🔍 Overview

Microbial growth in complex substrates often exhibits multiphasic (polyauxic) behavior, which standard monoauxic models (like Monod or First-Order) fail to capture accurately. This platform utilizes a **weighted sum of sigmoidal phases** to describe these dynamics, reformulating the canonical Boltzmann and Gompertz equations into biologically interpretable forms.

This tool bridges the gap between empirical curve fitting and mechanistic modeling by:

1. Extracting explicit kinetic parameters: **Maximum Specific Rate ()** and **Lag Phase ()**.
2. Employing a robust two-stage optimization pipeline to avoid local minima.
3. Automatically detecting statistical outliers using the ROUT method.

## 📐 Theoretical Framework

The platform models polyauxic growth as a summation of  distinct growth phases. The weighting factors () scale the contribution of each phase to the total amplitude .

### The Polyauxic Equation

The general form for the polyauxic model is:

$$ y(x) = y_i + (y_f - y_i) \sum_{j=1}^{n} p_j \cdot f_j(x, p_j) $$

Where  are weighting factors constrained by a Softmax transformation such that .

### 1. Modified Polyauxic Boltzmann

Based on the reparameterization of the Boltzmann equation (Eq. 31 in reference paper):

$$ y(x)=y_{i}+(y_{f}-y_{i})\cdot\sum_{j=1}^{n}\frac{p_{j}}{1+e^{\frac{4\cdot r_{max,j}\cdot(\lambda_{j}-x)}{(y_{f}-y_{i})\cdot p_{j}}+2}} $$

### 2. Modified Polyauxic Gompertz

Based on the reparameterization of the Gompertz equation for asymmetry (Eq. 32 in reference paper):

$$ y(x)=y_{i}+(y_{f}-y_{i})\cdot\sum_{j=1}^{n}p_{j}\cdot e^{-e^{\frac{r_{max,j}\cdot e}{(y_{f}-y_{i})\cdot p_{j}}(\lambda_{j}-x)+1}} $$

## ✨ Features

* **Heuristic Initialization:** Automatically estimates initial parameters (, ) by analyzing peaks in the first derivative of the input data to avoid arbitrary starting points.
* **Hybrid Optimization Pipeline:**
* **Global Search:** Uses **Differential Evolution (DE)** to navigate the high-dimensional, multimodal parameter space of polyauxic models.
* **Local Refinement:** Applies **L-BFGS-B** for precise convergence and strict bound enforcement (non-negative rates).


* **Robust Outlier Detection:** Implements the **ROUT method** (Robust Regression and Outlier Removal) using a Charbonnier loss function and False Discovery Rate (FDR) control to identify experimental anomalies without subjective bias.
* **Model Parsimony:** Automatically selects the optimal number of phases () using Information Criteria: **AIC**, **AICc** (for small sample sizes), and **BIC**.
* **Uncertainty Quantification:** Estimates standard errors via the Hessian matrix and propagates errors for derived parameters using the Delta method.

## 💻 Installation

To run the platform locally, you need Python 3.8+ installed.

1. **Clone the repository:**
```bash
git clone [https://github.com/gusmock/mono_polyauxic_kinetics.git](https://github.com/gusmock/mono_polyauxic_kinetics.git)
cd mono_polyauxic_kinetics

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*Required packages: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`.*

## 🚀 Usage

You can access the hosted web application or run it locally.

**Web Application:**
[https://monopolyauxickinetics-test.streamlit.app](https://monopolyauxickinetics-test.streamlit.app)

**Local Execution:**
Run the Streamlit app from your terminal:

```bash
streamlit run app.py

```

## 📂 Data Format

The application accepts `.csv` or `.xlsx` files. To ensure correct replicate detection, your data must be structured in **column pairs**.

* **Structure:** Time column followed immediately by Response column.
* **Replicates:** Add additional pairs of columns for biological replicates. The system automatically groups them.
* **Decimals:** Both dot (`.`) and comma (`,`) are accepted.

**Example Layout:**

| Time (Replica 1) | Response (Replica 1) | Time (Replica 2) | Response (Replica 2) |
| --- | --- | --- | --- |
| 0.0 | 0.105 | 0.0 | 0.102 |
| 1.0 | 0.200 | 1.0 | 0.198 |
| ... | ... | ... | ... |

## 📄 Citation

If you use this software or the underlying methodology in your research, please cite the following paper and the software DOI:

**Paper (ArXiv):**

> Mockaitis, G. (2025). Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics. *ArXiv*, 2507.05960. https://arxiv.org/abs/2507.05960

**Software (Zenodo):**

> Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0). *Zenodo*. https://www.google.com/url?sa=E&source=gmail&q=https://doi.org/10.5281/zenodo.18025828

**BibTeX:**

```bibtex
@article{mockaitis2025polyauxic,
  title={Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic Framework for Complex Biological Dynamics},
  author={Mockaitis, Gustavo},
  journal={ArXiv},
  volume={2507.05960},
  year={2025},
  url={[https://arxiv.org/abs/2507.05960](https://arxiv.org/abs/2507.05960)}
}

@software{mockaitis2025platform,
  author       = {Mockaitis, Gustavo},
  title        = {Polyauxic Modeling Platform (v1.0.0)},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18025828},
  url          = {[https://doi.org/10.5281/zenodo.18025828](https://doi.org/10.5281/zenodo.18025828)}
}

```

## 👨‍🔬 Author

**Prof. Dr. Gustavo Mockaitis**
*Interdisciplinary Research Group on Biotechnology Applied to the Agriculture and the Environment (GBMA)*
School of Agricultural Engineering, University of Campinas (FEAGRI/UNICAMP)
Campinas, SP, Brazil.
📧 gusmock@unicamp.br

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
