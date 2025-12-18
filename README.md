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
    git clone [https://github.com/your-username/polyauxic-modeling.git](https://github.com/your-username/polyauxic-modeling.git)
    cd polyauxic-modeling
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web App

To launch the graphical interface for single-dataset analysis:

```bash
streamlit run app.py
