# NASA Li-ion Battery RUL Prediction and SoH Classification

This project analyzes lithium-ion battery aging data from the NASA Ames Prognostics Center of Excellence (PCoE). The goal is to estimate battery Remaining Useful Life (RUL) and classify battery health state using per-cycle discharge features.

The final analysis is provided as a Jupyter notebook:

[`notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb`](notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb)

## Project Overview

Lithium-ion battery degradation is a key problem in electric vehicles, grid storage, and reliability engineering. This project uses NASA battery discharge-cycle data to build machine learning models for:

- RUL prediction: estimating cycles remaining before end-of-life.
- SoH classification: identifying whether a battery cycle is healthy or degraded.
- Feature analysis: understanding which discharge-cycle measurements contribute most to model performance.

## Dataset

The analysis is based on the NASA PCoE Li-ion Battery Aging Dataset. The dataset contains charge, discharge, and impedance measurements collected from batteries aged under controlled operating conditions.

Dataset source: [NASA Battery Dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

Raw dataset files are not included in this repository because they may be large and are better downloaded from the original source.

## Methodology

The notebook follows this workflow:

1. Load and clean NASA battery metadata and discharge-cycle files.
2. Extract cycle-level features from voltage, current, temperature, power, and time-series discharge data.
3. Compute State of Health (SoH), Remaining Useful Life (RUL), and degraded-health labels.
4. Detect anomalous battery profiles before modeling.
5. Train and evaluate machine learning models using battery-grouped validation.
6. Compare performance against a baseline CNN-LSTM result.
7. Export result tables and visualizations.

## Key Results

The rebuilt XGBoost approach improves over the original CNN-LSTM baseline under grouped battery-level validation:

| Metric | Original CNN-LSTM | Rebuilt XGBoost |
| --- | ---: | ---: |
| MAE (cycles) | 20.95 | 17.35 |
| RMSE (cycles) | 32.41 | 27.71 |
| R2 | -20.60 | 0.076 |

The main result is that engineered cycle-level features with gradient boosting provide a more stable and interpretable approach for this small battery dataset than a sequence-based deep learning model.

## Repository Structure

```text
mse643/
├── README.md
├── .gitignore
├── notebook/
│   ├── NASA_Battery_RUL_and_SoH_Classification.ipynb
│   └── requirements.txt
└── outputs/
    ├── 01_soh_trajectories.png
    ├── 02_correlation_heatmap.png
    ├── 03_target_distributions.png
    ├── 04_rul_actual_vs_pred.png
    ├── 05_feature_importance_reg.png
    ├── 06_classifier_eval.png
    ├── 07_feature_importance_clf.png
    ├── battery_profiles.csv
    ├── feature_table.csv
    └── summary_comparison.csv
```

## Outputs

The `outputs/` directory contains the generated figures and result tables from the final notebook:

- SoH degradation trajectories
- Feature correlation heatmap
- RUL and health-label distributions
- Actual vs predicted RUL plot
- Regression and classification feature importance plots
- Classification evaluation figure
- Processed feature table and summary comparison CSV

## Setup

Create a Python environment and install the required dependencies:

```bash
pip install -r notebook/requirements.txt
```

Then open and run the notebook:

```bash
jupyter notebook notebook/NASA_Battery_RUL_and_SoH_Classification.ipynb
```

## Notes

- The raw NASA dataset is not committed to this repository.
- The final notebook and generated outputs are included for review.
- The validation approach uses battery-grouped splitting to reduce leakage between training and test data.
