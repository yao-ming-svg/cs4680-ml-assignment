# main.py
# CS4680 Assignment 1 – Regression + Classification on one dataset
# Uses scikit-learn's California Housing to satisfy both tasks.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # no 'squared' kwarg
    rmse = mse ** 0.5                          # sqrt(MSE) = RMSE
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_classification(y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc}

def main():
    # 1) Load data
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y_reg = df["MedHouseVal"]

    # 2) Create classification label from the regression target
    threshold = y_reg.median()
    y_cls = (y_reg >= threshold).astype(int)

    # 3) Train/test split (use same split for both tasks; stratify by class)
    X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # 4) Preprocessing (all numeric; scaling helps linear/logistic)
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_features)],
        remainder="drop"
    )

    # 5) Regression models
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    # 6) Classification models
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    # 7) Fit + evaluate regression
    print("\n=== Regression Results (target: MedHouseVal) ===")
    reg_results = []
    for name, model in regressors.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, yreg_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_regression(yreg_test, y_pred)
        reg_results.append({"Model": name, **metrics})
    print(pd.DataFrame(reg_results).to_string(index=False))

    # 8) Fit + evaluate classification
    print("\n=== Classification Results (target: is_high_value) ===")
    cls_results = []
    for name, model in classifiers.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, ycls_train)
        # Predicted probabilities (if available); fallback to decision_function if needed
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # Convert decision scores to [0,1] via min-max for AUC fallback
            scores = pipe.decision_function(X_test)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            y_proba = scores
        y_pred = pipe.predict(X_test)
        metrics = evaluate_classification(ycls_test, y_proba, y_pred)
        cls_results.append({"Model": name, **metrics})
    print(pd.DataFrame(cls_results).to_string(index=False))

    # 9) Simple comparison guidance
    print("\nNote:")
    print("- Linear/Logistic benefit from scaling and provide interpretability.")
    print("- Random Forests capture non-linearities & interactions; often stronger accuracy but less interpretable.")
    print("- Discuss which is more suitable for your goal (value estimation vs quick segmenting).")

if __name__ == "__main__":
    main()
