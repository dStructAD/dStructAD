# dStructAD

a novel domain-level structured representation framework for unsupervised TSAD. dStructAD first builds structured domain-level normality based on the Kolmogorov--Arnold network, and then calibrates admissible variations via two phases. Phase~1 learns shared domain-level semi-structural normality knowledge from DomainSet, while Phase~2 performs structure-preserving TargetSet calibration to absorb admissible variations without collapsing anomaly separability.

## Project Structure

```text
dStructAD/
├─ data/                      # Datasets (place datasets here)
├─ data_provider/             # Data loading and dataset factory
│  ├─ __init__.py
│  ├─ data_factory.py         # Dataset registry / builder
│  └─ data_loader.py          # Dataset loading logic and preprocessing
├─ exp/                       # Experiment pipelines
│  ├─ __init__.py
│  ├─ exp_basic.py            # Base experiment class / shared training-eval utilities
│  └─ exp_2phase_dom_anomaly_detection.py
│                             # Main pipeline for two-phase domain-level TSAD
├─ model/                     # Model definitions
│  ├─ __init__.py
│  └─ st_kan.py               # Structural model (KAN-based) for dStructAD
├─ utils/                     # Utilities for evaluation and result management
│  ├─ affiliation/            # Supporting code for affiliation-based evaluation (if used)
│  ├─ vus/                    # VUS metrics implementation
│  ├─ __init__.py
│  ├─ evaluate_tools.py       # Metric computation and evaluation helpers
│  ├─ save_results.py         # Save metrics
│  └─ tools.py                # General helper functions
├─ run.py                     # Entry point script
└─ README.md                  # This file
```

 - **`run.py`** is the main function to launch the experiment.