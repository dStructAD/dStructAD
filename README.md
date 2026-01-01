# dStructAD

**dStructAD** is a novel **domain-level structured representation** framework for **unsupervised TSAD**.  
It first builds **structured domain-level normality** based on the **Kolmogorov–Arnold Network (KAN)**, and then **calibrates admissible variations** via a **two-phase** learning scheme:

- **Phase 1 (DomainSet)**: learns **shared, semi-structured** normality knowledge from **DomainSet**.  
- **Phase 2 (TargetSet)**: performs **structure-preserving** calibration on **TargetSet** to absorb **admissible variations** without collapsing **anomaly separability**.

---

## Project Structure

```text
dStructAD/
├─ data/                           # Datasets (place datasets here)
├─ data_provider/                  # Data loading and dataset factory
│  ├─ __init__.py
│  ├─ data_factory.py              # Dataset registry / builder
│  └─ data_loader.py               # Dataset loading logic and preprocessing
├─ exp/                            # Experiment pipelines
│  ├─ __init__.py
│  ├─ exp_basic.py                 # Base experiment class / shared training-eval utilities
│  └─ exp_2phase_dom_anomaly_detection.py
│                                  # Main pipeline for two-phase domain-level TSAD
├─ model/                          # Model definitions
│  ├─ __init__.py
│  └─ st_kan.py                    # Structural model (KAN-based) for dStructAD
├─ utils/                          # Utilities for evaluation and result management
│  ├─ affiliation/                 # Supporting code for affiliation-based evaluation (optional)
│  ├─ vus/                         # VUS metrics implementation
│  ├─ __init__.py
│  ├─ evaluate_tools.py            # Metric computation and evaluation helpers
│  ├─ save_results.py              # Save metrics and results
│  └─ tools.py                     # General helper functions
├─ run.py                          # Entry point script
└─ README.md                       # This file

```

## Run
 - **`run.py`** is the main function to launch the experiment.