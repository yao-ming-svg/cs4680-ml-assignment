## Overview
This repo trains two tasks on the same dataset:

1. **Regression** — predict `MedHouseVal` (median house value).  
2. **Binary classification** — predict `is_high_value` (1 if above the median value, else 0).

A shared train/test split and unified preprocessing enable apples-to-apples comparisons. Pipelines prevent leakage and keep code tidy.

## Problem Statement
- **Regression goal:** Estimate median home value  
- **Classification goal:** Flag neighborhoods as **high-value** vs **low-value**  
- **Why both?** Regression answers *how much?*; classification answers *which side of a policy-relevant threshold?*

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing` (no manual download).
- **Unit:** Each row is a California district with aggregated features and median home value.
- **Note:** Educational dataset; location variables may encode socio-economic patterns.

## Features & Targets

**Features (all numeric):**
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

**Targets:**
- **Regression:** `MedHouseVal` (continuous)
- **Classification:** `is_high_value` = 1 if `MedHouseVal` ≥ median, else 0

# Install Requirements
pip install -r requirements.txt

**or**

pip install scikit-learn pandas numpy

# How to run
python main.py

## Models

**Shared preprocessing**
- `StandardScaler` on all numeric columns (linear/logistic benefit; trees are scale-invariant but we keep one pipeline for clarity).

**Regression**
- `LinearRegression` — interpretable linear baseline  
- `RandomForestRegressor` — non-linear, captures interactions

**Classification**
- `LogisticRegression` — interpretable baseline with probabilities  
- `RandomForestClassifier` — robust non-linear tabular workhorse

## Evaluation

**Regression metrics**
- **MAE** (mean absolute error), **RMSE** (√MSE), **R²**

**Classification metrics**
- **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**

## Discussion & Suitability
- **Linear/Logistic:** great for interpretability and as a baseline when relationships are roughly linear (after scaling).  
- **Random Forests:** better when non-linearities and feature interactions matter (e.g., income × latitude/longitude).  
- **Task fit:** Regression supports pricing/forecasting; classification helps triage by value tiers.

## Limitations & Next Steps
- **No tuning yet:** Add `GridSearchCV`/`RandomizedSearchCV` (e.g., tree depth, leaf size, `n_estimators`; `C` for Logistic).  
- **Feature engineering:** Polynomial terms, interactions, or spatial features (e.g., coastal distance).  
- **Calibration & thresholds:** If deploying classification, consider `CalibratedClassifierCV` and cost-sensitive thresholds.  
- **Fairness:** Audit geographic proxies; assess disparate impact if used for real decisions.  
- **Try boosters (if allowed):** Gradient Boosting / XGBoost / LightGBM as additional baselines.

## Reproducibility Notes
- Fixed `random_state=42` for splits and forests.
- Suggested versions (see `requirements.txt`):
