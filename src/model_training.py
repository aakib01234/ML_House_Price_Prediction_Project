import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.data_preprocessing import FEATURES, TARGET, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "house_data.csv"
MODEL_DIR = ROOT / "models"

def mape(y_true, y_pred):
    denom = np.maximum(np.abs(np.asarray(y_true)), 1.0)
    return float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / denom)) * 100)

def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run python scripts/generate_data.py first.")
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    if len(df) < 30:
        raise ValueError("At least 30 labeled rows are required.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}),
        "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [120, 200, 300], "model__max_depth": [None, 12, 20],
             "model__min_samples_leaf": [1, 2, 4]}),
        "extra_trees": (ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [120, 200, 300], "model__max_depth": [None, 12, 20],
             "model__min_samples_leaf": [1, 2, 4]}),
        "gradient_boosting": (GradientBoostingRegressor(random_state=42),
            {"model__n_estimators": [100, 160, 240], "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [2, 3, 4]}),
    }
    results, fitted = {}, {}
    best_name, best_cv_mae, best_pipeline = None, float("inf"), None
    for name, (estimator, params) in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        search = RandomizedSearchCV(pipe, params, n_iter=min(8, int(np.prod([len(v) for v in params.values()]))),
                                   scoring="neg_mean_absolute_error", cv=5, random_state=42,
                                   n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        pred = search.predict(X_test)
        cv_mae = float(-search.best_score_)
        results[name] = {
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)),
            "test_mape_percent": mape(y_test, pred),
            "cv_mae_mean": cv_mae,
            "best_params": search.best_params_,
        }
        if cv_mae < best_cv_mae:
            best_name, best_cv_mae, best_pipeline = name, cv_mae, search.best_estimator_

    MODEL_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact = MODEL_DIR / f"house_price_{version}.joblib"
    joblib.dump(best_pipeline, artifact)
    current = {"version": version, "artifact": artifact.name, "algorithm": best_name,
               "trained_at_utc": datetime.now(timezone.utc).isoformat(), "features": FEATURES}
    (MODEL_DIR / "current.json").write_text(json.dumps(current, indent=2))
    (MODEL_DIR / f"metrics_{version}.json").write_text(json.dumps(results, indent=2))

    # Model-agnostic permutation importance on held-out data
    perm = permutation_importance(best_pipeline, X_test, y_test, n_repeats=8,
                                  random_state=42, scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std})
    importance.sort_values("importance_mean", ascending=False).to_csv(
        MODEL_DIR / f"feature_importance_{version}.csv", index=False)
    report = {"selected_model": best_name, "version": version, "dataset_rows": len(df),
              "train_rows": len(X_train), "test_rows": len(X_test), "results": results}
    (MODEL_DIR / f"report_{version}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return best_pipeline, results, current

if __name__ == "__main__":
    train()
