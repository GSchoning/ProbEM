# ProbEM: Stochastic Airborne Electromagnetic Inversion

**ProbEM** is a Python-based framework for the inversion of Airborne Electromagnetic (AEM) data. It is built upon the comprehensive simulation and inversion framework provided by **SimPEG** (Heagy et al., 2017) to provide both deterministic and stochastic interpretation of subsurface electrical conductivity.

The core of the repository is the **Randomized Maximum Likelihood (RML)** method, which acts as a "signal discovery device" to reveal sharp stratigraphic features and connectivity often smoothed out by conventional regularized inversions.

## Key Features

* **Stochastic Inversion (RML):** Generates an ensemble of plausible models to quantify uncertainty and highlight sharp conductivity contrasts.
* **Stochastic Depth of Investigation (DOI):** A data-driven approach using Singular Value Decomposition (SVD) and Signal-to-Noise Ratio (SNR) analysis to determine the reliable depth of the inversion.
* **Flexible Data Parsing:** Built-in support for .gex (SkyTEM) and .des / .dat (AEM standard) file formats.
* **SimPEG Integration:** Leverages SimPEG's adjoint solvers and optimization routines for high-fidelity electromagnetic simulations.

---

## Repository Structure

* `AEM_preproc.py`: Main preprocessing module managing `Survey` and `Data` classes.
* `ProbEM.py`: Core inversion logic containing the `Sounding`, `Calibration`, and `RML` classes.
* `libraries/`:
    * `gex_parser.py`: Parser for SkyTEM system configuration files.
    * `des_parser.py`: Parser for general AEM description files.

---

## Quick Start: Stochastic Workflow

1. System and Module Setup:

    import AEM_preproc
    import ProbEM
    import warnings
    import logging

    Survey = AEM_preproc.Survey
    Sounding = ProbEM.Sounding

    # Initialize Survey (SkyTEM example)
    survey = Survey()
    survey.proc_gex('path/to/system.gex')
    print("Modules reloaded.")

2. Run Stochastic Inversion:

    # Re-instantiate sounding for the specific line and time
    mock_sounding = Sounding(survey, mock_iline, mock_time, inv_thickness)

    # Suppress verbose Dask/Distributed logs
    warnings.filterwarnings('ignore')
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("dask").setLevel(logging.ERROR)

    # Generate realizations and execute
    n_realizations = 10
    mock_sounding.get_RML_reals(nreals=n_realizations)

    print(f"Running stochastic inversion with {n_realizations} realizations...")
    mock_sounding.RML.run_local()
    print("Stochastic inversion completed successfully.")

---

## Theory and Methodology

### 1. Layer Probability and Feature Discovery
Conventional regularized inversions provide Maximum A Posteriori (MAP) estimates of electrical conductivity (EC). While stable, these results often provide "subdued reflections" of conductivity contrasts, making it difficult to recognize offsets in aquitard connectedness.

**ProbEM** employs a feature-seeking methodology based on the Randomized Maximum Likelihood (RML) method:
* **EC Peaks/Troughs:** The ensemble of models is used to calculate the probability of a peak or trough of EC with depth.
* **Sharpness Filtering:** This acts as a high-amplitude sharpness filter, magnifying sub-horizontal EC contrasts rather than just EC magnitude, rendering features visible even under conductive cover.

### 2. Stochastic Depth of Investigation (DOI)
We define the DOI as the depth where the Signal-to-Noise Ratio (SNR) of the inversion drops below one. The workflow utilizes the inbuilt adjoint solver in **SimPEG** to compute the Jacobian ($J$), followed by Singular Value Decomposition (SVD).

The predictive error variance is computed for all possible truncation points ($k$) to define the optimal null-space cutoff. This yields a **Model Resolution Matrix (R)**, where each column represents the inversion's ability to resolve a feature at a specific depth.



**Probabilistic SNR Calculation:**
To move beyond a static DOI, we perform a Monte Carlo simulation to estimate the SNR probabilistically:
1.  **Noise Projection:** We generate thousands of random noise realizations based on data uncertainties and project them into the model space using the truncated SVD components.
2.  **Signal Comparison:** For each layer, we compare the theoretical "resolved signal" (derived from the diagonal of **R**) against the magnitude of this projected noise.
3.  **CDF Generation:** The DOI is defined for each realization as the depth where the resolved signal is overwhelmed by the projected noise (SNR < 1). By aggregating these results, we build a **Cumulative Distribution Function (CDF)** of the DOI, providing a robust, probabilistic measure of how deep the data reliably constrains the model.

---

## Development Context

This software was developed for the **Queensland Office of Groundwater Impact Assessment (OGIA)** and as part of a PhD research project at **Flinders University**, Australia.

## Contributors

* **Gerhard Schöning** (Lead Developer) — OGIA
* **John Doherty** — Watermark Numerical Computing
* **Sanjeev Pandey** — OGIA
* **Anna Bui Xuan Hy** — OGIA

## References

1. **Heagy, L. J., Cockett, R., Kang, S., Rosenkjaer, G. K., Heenan, L. J., & Oldenburg, D. W. (2017).** A framework for simulation and inversion in electromagnetics. *Computers & Geosciences*, 107, 1-19.
2. **SimPEG:** [https://simpeg.xyz/](https://simpeg.xyz/)

---
*Developed for the Queensland Office of Groundwater Impact Assessment (OGIA).*
