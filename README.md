# Polyauxic Modeling Platform & Experimental Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

A comprehensive computational tool for analyzing microbial growth kinetics, specifically designed to identify and model mono- and polyauxic behaviors. This repository contains both a user-friendly web application for direct analysis and a batch processing framework for large-scale statistical experiments.

## 📌 Features

* **Advanced Modeling:** Fits Boltzmann (Eq. 31) and Gompertz (Eq. 32) models reparameterized for biological interpretation.
* **Polyauxic Detection:** Automatically determines the number of growth phases (diauxic, triauxic, etc.) using Information Criteria (AIC, AICc, BIC).
* **Robust Statistics:** Implements ROUT (Robust regression + Outlier removal) with False Discovery Rate (FDR) control to handle noisy experimental data.
* **Global Optimization:** Uses Differential Evolution followed by L-BFGS-B refinement to escape local minima.
* **Interactive App:** Built with Streamlit for real-time visualization and data exploration.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or higher
* pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gusmock/mono_polyauxic_kinetics.git](https://github.com/gusmock/mono_polyauxic_kinetics.git)
    cd mono_polyauxic_kinetics
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web App

To launch the graphical interface for single-dataset analysis:

```bash
streamlit run app.py

```

---

## 🧪 Experimental Design

This repository includes a rigorous experimental framework (`run_experiment.py`) designed to benchmark model performance across diverse datasets. The goal is to statistically determine the optimal configuration for modeling polyauxic growth, adhering to the principle of parsimony (Occam's razor).

### Overview

We employ a **Full Factorial Design blocked by Dataset** to evaluate how different modeling factors influence the Information Criteria (AICc), fit quality (R^2_{adj}), and parameter uncertainty.

### Factors Evaluated

The experiment tests **360 unique conditions** (15 \times 2 \times 2 \times 3 \times 2) defined by the following factors:

| Factor | Type | Levels | Description |
| :--- | :--- | :--- | :--- |
| **Block** | Covariate | 15 Datasets | Categorized into three classes:<br>1. **First-order-like:** Smooth, simple decay/growth.<br>2. **Replicates:** Assays with biological variation.<br>3. **Unfinished:** Assays where the plateau is not fully reached. |
| **Factor A** | Model | **Boltzmann** vs. **Gompertz** | Compares symmetrical (Boltzmann) vs. asymmetrical (Gompertz) sigmoidal basis functions. |
| **Factor B** | Constraint | **Floating** vs. **Forced ($y_i=0$)** | Tests whether fixing the intercept to zero improves parameter precision or induces lack of fit. |
| **Factor C** | Robustness | **0.5%, 1.0%, 1.5%** (Q) | Varying the False Discovery Rate (FDR) in the ROUT outlier detection algorithm. |
| **Factor D** | Threshold | **Strict ($\Delta > 0$)** vs. **Conservative ($\Delta > 2$)** | The Information Criterion drop required to justify adding a new growth phase. $\Delta > 2$ adheres to Burnham & Anderson's rule for statistical distinguishability. |

### Methodology

1. **Batch Processing:** The script `run_experiment.py` iterates through all factor combinations for every dataset.
2. **Global Fitting:** For each combination, the algorithm fits models ranging from 1 to 5 phases.
3. **Model Selection:** The optimal number of phases (k_{opt}) is selected based on the **First Local Minimum** of the AICc, subject to the **Threshold (Factor D)** constraint.
4. **Data Collection:** The script exports a master CSV containing:
* Selected Number of Phases
* Goodness-of-fit metrics (R^2_{adj}, SSE)
* Information Criteria (AICc)
* Mean Standard Error (SE) of the rate parameters (r_{max})



### Running the Experiment

1. Place your datasets (CSV files) in the `datasets/` folder.
2. Map your files to their classes in `run_experiment.py`.
3. Execute the batch script:
```bash
python run_experiment.py

```


4. Analyze the resulting `final_experiment_results.csv` using ANOVA or paired t-tests.

---

## 📚 References & Theoretical Framework

The mathematical models and statistical methods implemented here are based on the following works:

**Primary Citation:**

* **Mockaitis, G. (2025).** *Mono and Polyauxic Growth Kinetic Models.* ArXiv: 2507.05960.

**Mathematical Basis:**

* **Boltzmann Model:** Adapted from Boltzmann, L. (1872). *Weitere Studien über das Wärmegleichgewicht unter Gasmolekülen.* (Eq. 31 in software).
* **Gompertz Model:** Adapted from Zwietering, M. H., et al. (1990). *Modeling of the bacterial growth curve.* Applied and Environmental Microbiology. (Eq. 32 in software).
* **Polyauxic Summation:** Mockaitis, G., et al. (2020). *Modeling polyauxic growth.* (Eq. 33 in software uses Softmax weighting).

**Statistical Methods:**

* **ROUT Method (Outliers):** Motulsky, H. J., & Brown, R. E. (2006). *Detecting outliers when fitting data with nonlinear regression – a new method based on robust nonlinear regression and the false discovery rate.* BMC Bioinformatics.
* **Differential Evolution:** Storn, R., & Price, K. (1997). *Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces.*
* **Information Criteria:** Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach.*

---

## 📂 Repository Structure

```text
.
├── app.py                 # Main Streamlit web application
├── polyauxic_lib.py       # Core mathematical library (The "Brain")
├── run_experiment.py      # Batch processing script for experimental design
├── requirements.txt       # Python dependencies
├── datasets/              # Folder for experimental CSV files
└── README.md              # Project documentation

```

## 📧 Contact

**Prof. Dr. Gustavo Mockaitis**

* **Affiliation:** School of Agricultural Engineering, University of Campinas (UNICAMP), Brazil.
* **Research Group:** GBMA (Interdisciplinary Research Group of Biotechnology Applied to the Agriculture and Environment)
* [Google Scholar Profile](https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=en&oi=ao)

```

```
