# SigNullEM: Stochastic Airborne Electromagnetic Inversion
![Uploading Gemini_Generated_Image_i3mcfli3mcfli3mc.png…]()


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![SimPEG](https://img.shields.io/badge/powered%20by-SimPEG-orange.svg)](https://simpeg.xyz)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SigNullEM** is a Python-based framework for the stochastic inversion of Airborne Electromagnetic (AEM) data. Built upon the **SimPEG** ecosystem (Heagy et al., 2017), it serves as a "signal discovery device," moving beyond deterministic smoothness to reveal the probability of subsurface structures.

The core philosophy of SigNullEM is **Randomized Maximum Likelihood (RML)**. By generating ensembles of plausible models rather than a single smooth estimate, this framework quantifies the reliability of stratigraphic boundaries and the true existence of deep conductors often obscured by the null space.

## Key Features

* **Probabilistic Feature Discovery:** Moves beyond visual interpretation of smooth models by explicitly calculating the probability of stratigraphic transitions (interfaces) and the presence of conductive units.
* **Non-Stationary Priors:** Implements depth-dependent covariance handling to correctly scale prior uncertainty against the physics of EM diffusion, preventing deep noise from being interpreted as structure.
* **Stochastic Inversion (RML):** Generates ensembles of models to quantify uncertainty and highlight sharp conductivity contrasts often invisible to deterministic smoothing.
* **Probabilistic Signal-to-Noise (SNR):** A robust Depth of Investigation (DOI) calculation that projects noise into the model null-space using Singular Value Decomposition (SVD).

---

## Repository Structure

* `AEM_preproc.py`: Main preprocessing module managing `Survey` and `Data` classes.
* `SigNullEM.py`: Core inversion logic containing the `Sounding`, `Calibration`, and `RML` classes.
* `libraries/`:
    * `gex_parser.py`: Parser for SkyTEM system configuration files.
    * `des_parser.py`: Parser for general AEM description files.

---

## Default Configuration

#### Python Definition
```python
import numpy as np

# Geometric layer thicknesses (m)
inv_thickness = np.array([
    2.0, 2.2, 2.4, 2.6, 2.8, 
    3.1, 3.4, 3.7, 4.0, 4.4, 
    4.8, 5.2, 5.6, 6.2, 6.7, 
    7.3, 8.0, 8.7, 9.5, 10.4, 
    11.3, 12.3, 13.4, 14.6, 16.0, 
    17.4, 19.0, 20.7, 22.6
])
```

---

## Quick Start: Stochastic Workflow

### 1. System and Module Setup

```python
import AEM_preproc
import SigNullEM
import numpy as np
import warnings
import logging

# Initialize classes
Survey = AEM_preproc.Survey
Sounding = SigNullEM.Sounding

# Load System Geometry
survey = Survey()
survey.proc_gex('path/to/system.gex')
print("System geometry loaded.")
```

### 2. Run Stochastic Inversion

```python
# Instantiate a sounding for a specific line (iline) and timestamp (time)
mock_sounding = Sounding(survey, mock_iline, mock_time, inv_thickness)

# Suppress verbose logs
warnings.filterwarnings('ignore')
logging.getLogger("distributed").setLevel(logging.ERROR)

# Generate realizations and execute
n_realizations = 50
print(f"Running stochastic inversion with {n_realizations} realizations...")

mock_sounding.get_RML_reals(nreals=n_realizations)
mock_sounding.RML.run_local()

print("Stochastic inversion completed.")
```

---

## Theory and Methodology

### 1. The Ill-Posed Inverse Problem
Airborne Electromagnetics is a fundamentally ill-posed problem; an infinite number of subsurface conductivity models can explain the observed data within noise limits. Conventional deterministic inversions (e.g., Occam's inversion) handle this by imposing **smoothness constraints**, selecting the "simplest" model that fits the data.

While stable, smooth models often suppress true geological complexity, smearing out sharp interfaces and subduing the magnitude of conductive anomalies. **SigNullEM** adopts a stochastic approach to explore the **null space**—the range of models that fit the data but differ in structure.

### 2. Randomized Maximum Likelihood (RML)
Instead of seeking a single optimal model, SigNullEM employs the **Randomized Maximum Likelihood (RML)** method.
1.  **Perturbation:** We generate an ensemble of starting models by perturbing the reference model and the observed data with random noise drawn from their respective covariance matrices.
2.  **Optimization:** Each realization is inverted independently. Because the objective function for each realization differs slightly, the resulting models explore the posterior probability distribution.
3.  **Outcome:** The ensemble captures the non-uniqueness of the problem. Where all models agree, the geology is well-constrained. Where they diverge, the data lacks sensitivity.

### 3. Non-Stationary Priors
Geological complexity and EM sensitivity are not uniform with depth. SigNullEM utilizes **non-stationary priors**, meaning the regularization constraints (covariance) are not static but vary as a function of depth.
* **Physics-Aware Scaling:** As the EM field diffuses, resolution naturally decays. Our priors loosen at depth to reflect this increased uncertainty, preventing the inversion from over-penalizing deep structures that the data can barely "see."
* **Structural Adaptation:** This allows the inversion to support sharp transitions in the near-surface (where signal is high) while gracefully admitting higher uncertainty in the basement.

### 4. Probabilistic Discovery of Transitions
The resulting ensemble allows us to move from "images" to "probabilities." By analyzing the derivatives of the conductivity profiles across the entire ensemble, we can calculate:
* **Transition Probability:** The likelihood of a lithological boundary (sharp change in conductivity) existing at a specific depth.
* **Conductor Probability:** The probability that a specific zone exceeds a conductivity threshold, distinguishing robust targets from inversion artifacts.

### 5. Stochastic Depth of Investigation (DOI)
We define the DOI as the depth where the Signal-to-Noise Ratio (SNR) of the inversion drops below 1.0. This is calculated via a Monte Carlo simulation where random noise is projected into the model space using the truncated Singular Value Decomposition (SVD) of the Jacobian. This yields a **Cumulative Distribution Function (CDF)** of the DOI, providing a probabilistic limit to interpretation.

---

## Development Context

This software was developed for the **Queensland Office of Groundwater Impact Assessment (OGIA)** and as part of a PhD research project at **Flinders University**, Australia.

## Contributors

* **Gerhard Schöning** (Lead Developer) — OGIA
* **John Doherty** — Watermark Numerical Computing
* **Sanjeev Pandey** — OGIA
* **Anna Bui Xuan Hy** — OGIA

## References

**SigNullEM** is the accompanying code for the following research (currently in preparation):

1.  **Schöning, G., Doherty, J., Pandey, S., & Hy, A. B. X. (2025).** *A stochastic framework for signal discovery and uncertainty quantification in Airborne Electromagnetic inversion.* (Paper in preparation).

It also builds upon the following foundational frameworks:

2.  **Heagy, L. J., Cockett, R., Kang, S., Rosenkjaer, G. K., Heenan, L. J., & Oldenburg, D. W. (2017).** A framework for simulation and inversion in electromagnetics. *Computers & Geosciences*, 107, 1-19.
3.  **Blatter, D., Key, K., & Ray, A. (2018).** Two-dimensional Bayesian inversion of electromagnetic data with a trans-dimensional Gaussian process prior. *Geophysical Journal International*.
4.  **SimPEG:** [https://simpeg.xyz/](https://simpeg.xyz/)
