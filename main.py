# main.py
# CS4680 Assignment 1 – Regression + Classification on California Housing

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
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix
)

# Pretty Printing
def dollars_from_units(x_units: float) -> str:
    """Convert housing units (100k USD) to a $ string."""
    return f"${x_units * 100_000:,.0f}"

def dollars_value(x_value: float) -> str:
    """Same as above but for raw $ values (already in dollars)."""
    return f"${x_value:,.0f}"

def pct(x: float) -> str:
    return f"{100 * x:.1f}%"

def top_k(series: pd.Series, k: int = 5) -> pd.Series:
    return series.sort_values(ascending=False).head(k)

# Metric Wrappers
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5   # manual sqrt for wide compat
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_classification(y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_proba)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc}

def explain_regression(df_reg, y_test):
    avg_price_dollars = y_test.mean() * 100_000
    med_price_dollars = y_test.median() * 100_000

    print("\n=== Regression Results (target: MedHouseVal) ===")
    print(df_reg.to_string(index=False))

    # Pick best by R²
    best_row = df_reg.sort_values("R2", ascending=False).iloc[0]
    mae_dollars = dollars_from_units(best_row.MAE)
    rmse_dollars = dollars_from_units(best_row.RMSE)
    rel_mae = best_row.MAE / y_test.mean()
    rel_rmse = best_row.RMSE / y_test.mean()

    print("\nWhat this means (Regression):")
    print(
        f"- The average test-set home value is about {dollars_value(avg_price_dollars)} "
        f"(median {dollars_value(med_price_dollars)})."
    )
    print(
        f"- Using {best_row.Model}, the typical absolute error (MAE) is {mae_dollars} "
        f"(~{pct(rel_mae)} of the average price)."
    )
    print(f"- The RMSE is {rmse_dollars} (penalizes large misses more).")
    print(f"- R² = {best_row.R2:.3f} → the model explains ~{pct(best_row.R2)} of the price variation.")

def explain_classification(df_cls, y_true, y_pred_map, y_proba_map, label="HIGH value (>= median)"):
    # Class balance
    counts = pd.Series(y_true).value_counts()
    pos_share = counts.get(1, 0) / counts.sum()

    print("\n=== Classification Results (target: is_high_value) ===")
    print(df_cls.to_string(index=False))

    # Choose best by F1 (or ROC-AUC as a tie-breaker)
    df_cls = df_cls.sort_values(["F1", "ROC_AUC", "Accuracy"], ascending=False)
    best_name = df_cls.iloc[0]["Model"]

    y_pred_best = y_pred_map[best_name]
    y_proba_best = y_proba_map[best_name]
    cm = confusion_matrix(y_true, y_pred_best, labels=[0,1])

    # Extract key numbers
    metrics = df_cls.iloc[0]
    prec, rec = metrics["Precision"], metrics["Recall"]

    print("\nClass balance (test set):")
    print(f"- {pct(pos_share)} are {label}; {pct(1 - pos_share)} are NOT {label}.")

    print("\nConfusion matrix (rows = actual, cols = predicted) [0=LOW, 1=HIGH]:")
    print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    print("\nWhat this means (Classification):")
    print(f"- Using {best_name}, Accuracy is {pct(metrics['Accuracy'])} and ROC-AUC is {metrics['ROC_AUC']:.3f}.")
    print(f"- Precision {pct(prec)} → if the model flags 100 neighborhoods as {label}, "
          f"~{pct(prec)} of them are truly {label}.")
    print(f"- Recall    {pct(rec)} → out of 100 truly {label} neighborhoods, "
          f"the model correctly catches ~{pct(rec)}.")

def feature_explanations(reg_pipes, cls_pipes, feature_names):
    print("\n=== Feature Signals (why the models decided what they did) ===")

    # Random Forest Regressor importances
    if "RandomForestRegressor" in reg_pipes:
        rf_reg = reg_pipes["RandomForestRegressor"]
        if hasattr(rf_reg.named_steps["model"], "feature_importances_"):
            imp = pd.Series(rf_reg.named_steps["model"].feature_importances_, index=feature_names)
            print("\nTop features for RandomForestRegressor (importance):")
            print(top_k(imp, 5).round(3).to_string())

    # Linear Regression coefficients (on standardized features)
    if "LinearRegression" in reg_pipes:
        lin = reg_pipes["LinearRegression"]
        coefs = pd.Series(lin.named_steps["model"].coef_, index=feature_names).abs()
        print("\nLargest-magnitude coefficients for LinearRegression (after scaling):")
        print(top_k(coefs, 5).round(3).to_string())

    # Random Forest Classifier importances
    if "RandomForestClassifier" in cls_pipes:
        rf_cls = cls_pipes["RandomForestClassifier"]
        if hasattr(rf_cls.named_steps["model"], "feature_importances_"):
            imp = pd.Series(rf_cls.named_steps["model"].feature_importances_, index=feature_names)
            print("\nTop features for RandomForestClassifier (importance):")
            print(top_k(imp, 5).round(3).to_string())

    # Logistic Regression coefficients
    if "LogisticRegression" in cls_pipes:
        logi = cls_pipes["LogisticRegression"]
        coef = logi.named_steps["model"].coef_.ravel()
        coefs = pd.Series(np.abs(coef), index=feature_names)
        print("\nLargest-magnitude coefficients for LogisticRegression (after scaling):")
        print(top_k(coefs, 5).round(3).to_string())

def main():
    # Load data
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y_reg = df["MedHouseVal"]  # in 100k USD

    # Create classification label from the regression target
    threshold = y_reg.median()
    y_cls = (y_reg >= threshold).astype(int)

    # Train/test split (use same split for both tasks; stratify by class)
    X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # 4) Preprocessing (all numeric; scaling helps linear/logistic)
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_features)],
        remainder="drop"
    )

    # Build models
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    # Fit & evaluate regression, keeping fitted pipelines for explanations
    reg_results = []
    reg_pipes = {}
    for name, model in regressors.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, yreg_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_regression(yreg_test, y_pred)
        reg_results.append({"Model": name, **metrics})
        reg_pipes[name] = pipe
    df_reg = pd.DataFrame(reg_results)

    # Fit & evaluate classification (+ probabilities), keep fitted pipelines
    cls_results = []
    y_pred_map, y_proba_map = {}, {}
    cls_pipes = {}
    for name, model in classifiers.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, ycls_train)

        # Predicted probabilities for the positive class (1)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # Fallback: scale decision_function scores to [0,1] (for AUC only)
            scores = pipe.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        y_pred = pipe.predict(X_test)

        metrics = evaluate_classification(ycls_test, y_proba, y_pred)
        cls_results.append({"Model": name, **metrics})
        y_pred_map[name] = y_pred
        y_proba_map[name] = y_proba
        cls_pipes[name] = pipe
    df_cls = pd.DataFrame(cls_results)

    # explanations
    explain_regression(df_reg, yreg_test)
    explain_classification(df_cls, ycls_test, y_pred_map, y_proba_map)

    # Brief feature explanations (top signals)
    feature_explanations(reg_pipes, cls_pipes, numeric_features)

    print("\nNote:")
    print("- Prices are reported in US dollars; the original target is in units of $100,000.")
    print("- Linear/Logistic = interpretable baselines; Random Forests capture non-linearities & interactions.")
    print("- Use this output block as your README 'Results' + 'Discussion' text.")

if __name__ == "__main__":
    main()
