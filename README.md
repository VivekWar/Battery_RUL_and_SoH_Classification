# NASA Li-ion Battery — RUL Prediction & State-of-Health Classification

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-017CEE)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-success)

Predicting the **Remaining Useful Life (RUL)** and classifying the **State of Health (SoH)**
of lithium-ion cells from the NASA Ames Prognostics Center of Excellence (PCoE) battery
aging dataset. The project engineers per-cycle features from raw discharge traces and
trains gradient-boosted models under **honest, leave-batteries-out validation**, then
benchmarks them against the original CNN-LSTM deep-learning approach and a linear baseline.

> **Deliverable:** [`notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb`](notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Prediction Targets](#prediction-targets)
- [Modeling & Validation](#modeling--validation)
- [Repository Structure](#repository-structure)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Lithium-ion battery degradation is a central problem in electric vehicles, grid storage,
consumer electronics, and reliability engineering. Accurately estimating how many cycles
a cell has left — and flagging when it has crossed into a degraded state — enables
predictive maintenance and safe end-of-life management.

This project addresses two tasks on the NASA PCoE dataset:

1. **RUL prediction (regression)** — estimate the number of discharge cycles remaining
   before a cell reaches End-of-Life (`SoH < 0.75`).
2. **Health-status classification** — label each cycle as **Healthy** or **Degraded**
   (`SoH < 0.80`).

The core methodological argument is that **per-cycle tabular feature engineering + gradient
boosting** is a more stable, interpretable, and reproducible approach for this small dataset
than raw-sequence deep learning — *once both models are evaluated fairly* with grouped,
leave-batteries-out cross-validation that prevents cycles from the same cell leaking across
the train/test boundary.

---

## Key Results

All models are evaluated with **5-fold `GroupKFold`** (batteries held out entirely per fold).
Metrics are reported as mean ± standard deviation across the five outer folds; **bold** marks
the best value per row.

| Metric | Original CNN-LSTM<br>(`ShuffleSplit`, leaked) | CNN-LSTM<br>(re-validated, `GroupKFold`) | Linear / Logistic<br>baseline | **This work**<br>(nested-tuned XGBoost) |
|---|---|---|---|---|
| RUL MAE (cycles) ↓ | 20.95 | 22.85 ± 16.35 | 24.95 ± 8.89 | **16.76 ± 13.88** |
| RUL RMSE (cycles) ↓ | 32.41 | 31.24 ± 21.76 | — | **24.46 ± 18.47** |
| RUL R² ↑ | −20.60 | −0.51 ± 1.60 | — | **0.243 ± 0.651** |
| Classifier Accuracy ↑ | — | — | 0.787 ± 0.177 | **0.819 ± 0.077** |
| Classifier F1 (Degraded) ↑ | — | — | **0.653 ± 0.324** | 0.602 ± 0.299 |
| Classifier ROC-AUC ↑ | — | — | 0.953 ± 0.063 | **0.960 ± 0.051** |

**Takeaways**

- **Validation design matters more than model choice.** The original CNN-LSTM's headline
  `ShuffleSplit` result (R² = −20.6) was partly a data-leakage artifact — but re-training the
  *same* architecture under fair `GroupKFold` still yields R² = −0.51, so raw-sequence deep
  learning is a genuinely poor fit for ~26 batteries, not merely a victim of a leaky split.
- **XGBoost recovers a positive R²** with orders of magnitude fewer parameters and no
  GPU-scale training, and cuts RUL MAE by ~33% versus a plain linear baseline (24.95 → 16.76).
- **The classification task is well-posed and near-linearly separable.** XGBoost's edge over
  logistic regression is marginal there (ROC-AUC 0.960 vs 0.953), so the added model
  complexity mostly pays off on the harder regression task.

> ℹ️ The exported [`outputs/summary_comparison.csv`](outputs/summary_comparison.csv) is a
> snapshot from an earlier run; exact figures vary slightly between runs because
> hyperparameters are selected by a randomized search. The table above reflects the
> notebook's final reported summary.

---

## Dataset

- **Source:** NASA PCoE Li-ion Battery Aging Dataset
  ([Kaggle mirror](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)).
- **Contents:** charge, discharge, and impedance measurements collected from lithium-ion
  cells cycled to failure under controlled operating conditions. This project uses the
  **discharge** cycles only.
- **Scale:** ~34 batteries in the raw metadata; 34 have usable discharge profiles; **24 are
  retained** for modeling after Isolation Forest outlier removal.

Each discharge cycle provides time-series traces of `Voltage_measured`, `Current_measured`,
`Temperature_measured`, and `Time`, plus a per-cycle `Capacity`.

> **Note:** the raw dataset is **not committed** to this repository (it is large and best
> obtained from the original source). Place the extracted data locally and point the notebook
> at it — see [Usage](#usage).

---

## Methodology

The notebook follows an end-to-end pipeline:

1. **Load & Clean** — parse the metadata index and filter to valid discharge cycles.
2. **Feature Engineering** — extract 20 scalar features from raw V/I/T traces, visualize raw physical telemetry, apply targeted single-cycle capacity spike correction, and flag anomalous battery aging profiles with an Isolation Forest.
3. **Targets: SoH, RUL, Degraded** — derive per-cycle health labels and BMS-realistic trend features.
4. **EDA (Targets & Health)** — SoH degradation trajectories, feature correlations, and target distributions.
5. **Modelling Utilities** — shared helpers for the nested cross-validation loops.
6. **RUL Regression** — nested `GroupKFold` randomized hyperparameter search for an XGBoost regressor, benchmarked against a CNN-LSTM sequence model and linear baseline, alongside residual analysis, ablation, and learning curves.
7. **Healthy vs Degraded Classifier** — nested `GroupKFold` search for an XGBoost classifier, benchmarked against a logistic baseline, plus probability calibration checks.
8. **Inference Demo** — end-to-end cycle scoring using models persisted to `artifacts/models/`.
9. **Summary** — tabular metric comparison and final key takeaways.

---

## Feature Engineering

This stage consists of three key steps to transform raw physical telemetry into a robust, clean dataset:

### 1. Cycle-Level Feature Extraction
Each discharge cycle is summarized into a fixed-length vector. **20 features** are used for
classification; the regressor adds **3 cumulative trend features** (23 total).

| Group | Features |
|---|---|
| **Voltage** | `v_mean`, `v_min`, `v_max`, `v_std` |
| **Current** | `c_mean`, `c_min`, `c_std` |
| **Temperature** | `temp_mean`, `temp_max`, `temp_std` |
| **Power** | `power_mean`, `power_std` |
| **Voltage slope (dV/dt)** | `dv_dt_mean`, `dv_dt_std` |
| **Cycle integrals** | `discharge_time`, `charge_throughput`, `energy`, `n_samples` |
| **Context** | `ambient_temperature`, `cycle_idx` |
| **Trend (regressor only)** | `capacity_so_far`, `soh_so_far`, `fade_rate_recent` |

The classifier deliberately excludes any capacity-derived feature — capacity is the source of
the label, so including it would leak the target.

### 2. Targeted Spike Correction
Several batteries exhibit physically impossible single-cycle capacity spikes (e.g., recovering 30% capacity in one cycle, then losing it the next) due to sensor noise. These localized anomalies are identified via z-score thresholds on rolling capacity differences and replaced using linear interpolation to preserve real degradation trends.

### 3. Automated Battery Quality Filter
After spike correction, some battery capacity curves remain fundamentally unreliable. The pipeline engineers curve-quality features per battery (`max_jump`, `std_diff`, `n_large_jumps`, `monotonicity`) and uses an **Isolation Forest** (`contamination=0.30`) to automatically flag and drop 10 unreliable profiles, leaving 24 clean batteries for training.

---

## Prediction Targets

All targets are computed **per battery**, relative to that cell's own first-cycle capacity.

| Target | Definition |
|---|---|
| **SoH** | `Capacity / first_cycle_Capacity` (State of Health as a ratio) |
| **RUL** | Cycles until the first cycle where `SoH < 0.75` (End-of-Life), floored at 0 afterward |
| **Degraded** | `1` if `SoH < 0.80`, else `0` (an earlier warning line than EOL) |

---

## Modeling & Validation

- **Estimators:** `XGBRegressor` (RUL) and `XGBClassifier` (health status).
- **Validation:** **nested `GroupKFold`** — the outer loop holds out entire batteries for
  testing, and the inner `RandomizedSearchCV` (also grouped) selects hyperparameters without
  ever seeing the held-out cell. No cycle from a test battery appears in training.
- **Baselines:** `LinearRegression` / `LogisticRegression` on identical folds.
- **Deep-learning benchmark:** a CNN-LSTM over raw cycle sequences (`SEQ_LEN = 10`,
  `CNN_EPOCHS = 30`), re-validated under the same `GroupKFold` protocol for a fair comparison.
- **Outlier removal:** `IsolationForest` (`contamination = 0.30`) applied to engineered curve-quality features per battery.
- **Burn-in:** the first `MIN_CYCLES_TO_EVAL = 10` cycles per battery are excluded from evaluation.

---

## Repository Structure

```text
mse643/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── cleaned_dataset/           # local dataset (symlink; not committed)
├── notebook/
│   ├── NASA_Battery_RUL_and_SoH_Classification.ipynb   # main deliverable
│   ├── requirements.txt           # pinned dependencies
│   └── artifacts/models/          # trained models (generated at runtime; not committed)
└── outputs/
    ├── 01_soh_trajectories.png
    ├── 02_correlation_heatmap.png
    ├── 03_target_distributions.png
    ├── 04_rul_actual_vs_pred.png
    ├── 05_feature_importance_reg.png
    ├── 06_classifier_eval.png
    ├── 07_feature_importance_clf.png
    ├── battery_profiles.csv        # per-battery aging profiles + outlier flags
    ├── feature_table.csv           # full engineered per-cycle feature table
    └── summary_comparison.csv      # exported metric comparison snapshot
```

---

## Outputs

The `outputs/` directory holds the generated figures and result tables:

| File | Description |
|---|---|
| `01_soh_trajectories.png` | SoH degradation curves across cells |
| `02_correlation_heatmap.png` | Feature correlation heatmap |
| `03_target_distributions.png` | RUL and health-label distributions |
| `04_rul_actual_vs_pred.png` | Actual vs. predicted RUL |
| `05_feature_importance_reg.png` | Regressor feature importances |
| `06_classifier_eval.png` | Classification evaluation (confusion matrix / ROC) |
| `07_feature_importance_clf.png` | Classifier feature importances |
| `battery_profiles.csv` | Per-battery aging summary with Isolation Forest anomaly flags |
| `feature_table.csv` | Full engineered feature table (per cycle) |
| `summary_comparison.csv` | Metric comparison across models |

---

## Installation

Requires **Python 3.10**. Create an isolated environment and install the pinned dependencies:

```bash
# clone
git clone <repository-url>
cd mse643

# create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r notebook/requirements.txt
```

---

## Usage

1. **Obtain the dataset** from the [source](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
   and extract it under `data/` so the layout is:

   ```text
   data/cleaned_dataset/
   ├── metadata.csv
   └── data/            # per-cycle discharge CSV files
   ```

2. **Run the notebook from the repository root:**

   ```bash
   jupyter notebook notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb
   ```

   It will build the feature table, train and evaluate both models, produce the figures in
   `outputs/`, and persist the trained models to `notebook/artifacts/models/`.

   The paths default to the repo-relative layout above and need no editing when Jupyter is
   launched from the project root:

   ```python
   BASE_DIR      = os.environ.get("BATTERY_DATA_DIR", "data/cleaned_dataset")
   METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
   DATA_DIR      = os.path.join(BASE_DIR, "data")
   ARTIFACT_DIR  = os.environ.get("BATTERY_ARTIFACT_DIR", "notebook/artifacts")
   ```

   For a container or CI run with data mounted elsewhere, override via environment variables:

   ```bash
   BATTERY_DATA_DIR=/data BATTERY_ARTIFACT_DIR=/scripts/artifacts \
     jupyter nbconvert --to notebook --execute \
     notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb
   ```

---

## Reproducibility

- A **single seed** (`RNG = 42`) drives NumPy, TensorFlow, and every estimator.
- **Dependencies are pinned** in [`notebook/requirements.txt`](notebook/requirements.txt);
  package versions are also logged at runtime.
- **Trained models are persisted** (regressor and classifier, each with its scaler) to
  `artifacts/models/*.joblib` for reuse without retraining.

---

## Limitations

- Only **24–34 batteries** are available after cleaning, which caps how well any model can
  extrapolate a full lifespan from a few early cycles — this, not model choice, is the main
  ceiling on RUL R².
- **2 of 5 outer folds** still show negative R² even for XGBoost: their held-out cells exhibit
  fade patterns the training set does not cover.
- The **classification task is better-posed** than regression and correspondingly more reliable.

---

## Future Work

- **Quantile / interval regression** for calibrated RUL uncertainty bounds.
- **Per-battery calibration curves** for the health classifier.
- **More batteries or transfer learning** to test whether the R² ceiling is truly data-limited.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data | NumPy, pandas, SciPy |
| Modeling | XGBoost, scikit-learn |
| Deep-learning benchmark | TensorFlow / Keras |
| Visualization | Matplotlib, seaborn |
| Persistence / notebook | joblib, Jupyter, nbformat |

---

## Acknowledgments

- **NASA Ames Prognostics Center of Excellence (PCoE)** for the Li-ion Battery Aging Dataset.
- Dataset mirror courtesy of the [Kaggle community](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset).

---

## License

Released under the [MIT License](LICENSE). Developed as academic coursework (MSE 643).
