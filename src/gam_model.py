from pathlib import Path
import gc
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pygam import LinearGAM, s, f

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_earners.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_ORDER = [
    "age",
    "formal_education_years",
    "gender",
    "employment_status",
    "industry",
    "occupation",
]

CATEGORICAL_FEATURES = {"gender", "employment_status", "industry", "occupation"}
CV_FOLDS = 5
CV_MAX_ROWS = 10_000
RANDOM_STATE = 42


def load_and_prepare_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "clean_earners.csv not found. Run notebooks/01_data_preparation.ipynb first."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURE_ORDER + ["earnings"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for GAM: {missing}")

    model_df = df[required].copy()
    for column in required:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df[model_df["gender"].isin([1, 2])]
    return model_df.dropna().astype(np.float32, copy=False)


def make_baseline_row(reference_df: pd.DataFrame) -> np.ndarray:
    values = []
    for feature in FEATURE_ORDER:
        if feature in CATEGORICAL_FEATURES:
            values.append(float(reference_df[feature].mode().iloc[0]))
        else:
            values.append(float(reference_df[feature].median()))
    return np.asarray(values, dtype=float)


def save_gam_effects(gam: LinearGAM, reference_df: pd.DataFrame) -> Path:
    baseline = make_baseline_row(reference_df)
    effect_rows = []

    smooth_features = {0: "Age", 1: "Formal education years"}
    for term_index, feature_name in smooth_features.items():
        feature = FEATURE_ORDER[term_index]
        values = np.linspace(
            reference_df[feature].quantile(0.01),
            reference_df[feature].quantile(0.99),
            100,
        )
        grid = np.tile(baseline, (len(values), 1))
        grid[:, term_index] = values

        effect, confidence = gam.partial_dependence(
            term=term_index, X=grid, width=0.95
        )

        for value, effect_value, lower, upper in zip(
            values, effect, confidence[:, 0], confidence[:, 1]
        ):
            effect_rows.append(
                {
                    "feature": feature_name,
                    "value": float(value),
                    "effect": float(effect_value),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )

    factor_features = {
        2: "Gender",
        3: "Employment status",
        4: "Industry",
        5: "Occupation",
    }

    for term_index, feature_name in factor_features.items():
        feature = FEATURE_ORDER[term_index]
        values = np.sort(reference_df[feature].unique())
        grid = np.tile(baseline, (len(values), 1))
        grid[:, term_index] = values

        effect, confidence = gam.partial_dependence(
            term=term_index, X=grid, width=0.95
        )

        for value, effect_value, lower, upper in zip(
            values, effect, confidence[:, 0], confidence[:, 1]
        ):
            effect_rows.append(
                {
                    "feature": feature_name,
                    "value": float(value),
                    "effect": float(effect_value),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )

    temporary_path = OUTPUT_DIR / "gam_effects.csv.tmp"
    pd.DataFrame(effect_rows).to_csv(temporary_path, index=False)
    return temporary_path


def save_gender_predictions(gam: LinearGAM, reference_df: pd.DataFrame) -> Path:
    baseline = make_baseline_row(reference_df)
    gender_grid = np.tile(baseline, (2, 1))
    gender_grid[:, FEATURE_ORDER.index("gender")] = [1, 2]
    predictions = gam.predict(gender_grid)
    temporary_path = OUTPUT_DIR / "gam_gender_predictions.csv.tmp"
    predictions_df = pd.DataFrame(
        {
            "gender": [1, 2],
            "gender_label": ["Male", "Female"],
            "predicted_earnings": predictions,
        }
    )
    predictions_df.to_csv(temporary_path, index=False)
    return temporary_path


def build_gam() -> LinearGAM:
    """Create the GAM specification used consistently for CV and final fitting."""
    return LinearGAM(
        s(0, n_splines=18)
        + s(1, n_splines=18)
        + f(2)
        + f(3)
        + f(4)
        + f(5)
    )


def cross_validate_gam(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Evaluate the GAM using folds made only from the training partition.

    The final test partition is intentionally not passed to this function. Each
    fold fits a fresh GAM on four training folds and evaluates it on the fifth,
    so these scores estimate validation performance without test-set leakage.
    If full-data CV exceeds the available WSL memory, a reproducible representative
    subset of at most ``CV_MAX_ROWS`` is sampled from the training partition only.
    This memory fallback affects CV estimation only; the final GAM still fits all
    rows in the 80% training partition.
    """
    if len(X_train) > CV_MAX_ROWS:
        sampled_indices = X_train.sample(
            n=CV_MAX_ROWS, random_state=RANDOM_STATE
        ).index
        cv_X = X_train.loc[sampled_indices]
        cv_y = y_train.loc[sampled_indices]
        print(
            f"CV memory fallback: using {len(cv_X):,} representative training rows "
            f"from {len(X_train):,}; final fit remains full-size.",
            flush=True,
        )
    else:
        cv_X = X_train
        cv_y = y_train

    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {"r2": [], "rmse": [], "mae": []}

    for fold_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(cv_X), start=1
    ):
        fold_start = time.perf_counter()
        print(f"CV fold {fold_number}/{CV_FOLDS} starting...", flush=True)
        fold_gam = build_gam()
        try:
            fold_gam.fit(
                cv_X.iloc[fit_indices].values,
                cv_y.iloc[fit_indices].values,
            )
            validation_predictions = fold_gam.predict(
                cv_X.iloc[validation_indices].values
            )
            validation_actual = cv_y.iloc[validation_indices]
            fold_scores["r2"].append(
                r2_score(validation_actual, validation_predictions)
            )
            fold_scores["rmse"].append(
                np.sqrt(mean_squared_error(validation_actual, validation_predictions))
            )
            fold_scores["mae"].append(
                mean_absolute_error(validation_actual, validation_predictions)
            )
            elapsed = time.perf_counter() - fold_start
            print(
                f"CV fold {fold_number}/{CV_FOLDS} completed in {elapsed:.1f}s.",
                flush=True,
            )
        finally:
            # pyGAM stores fitted matrices on the model. Release each fold
            # before constructing the next one instead of retaining six models.
            del fold_gam
            gc.collect()

    metrics = {
        "cv_r2_mean": float(np.mean(fold_scores["r2"])),
        "cv_r2_std": float(np.std(fold_scores["r2"], ddof=1)),
        "cv_rmse_mean": float(np.mean(fold_scores["rmse"])),
        "cv_rmse_std": float(np.std(fold_scores["rmse"], ddof=1)),
        "cv_mae_mean": float(np.mean(fold_scores["mae"])),
        "cv_mae_std": float(np.std(fold_scores["mae"], ddof=1)),
        "cv_training_rows": int(len(cv_X)),
        "cv_used_representative_subset": bool(len(cv_X) < len(X_train)),
    }
    del cv_X, cv_y, splitter, fold_scores
    gc.collect()
    return metrics


def train_and_evaluate_gam(model_df: pd.DataFrame) -> dict:
    """Run leakage-safe validation and final evaluation for the analytical GAM.

    First, an 80/20 split reserves an untouched final test partition. Five-fold
    cross-validation is then performed only on the 80% training partition.
    After CV, a final GAM is fit on all training rows and evaluated once on the
    reserved test rows. Predictions are evaluation tools; the project's goal is
    interpreting adjusted earnings associations and gender differences.
    """
    X = model_df[FEATURE_ORDER]
    y = model_df["earnings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(
        f"Reserved final test set: {len(X_test):,} rows. "
        f"Training partition: {len(X_train):,} rows."
    )
    print(f"Running {CV_FOLDS}-fold CV on the training partition only...")
    cv_metrics = cross_validate_gam(X_train, y_train)
    print(
        "CV summary: " + json.dumps(cv_metrics, sort_keys=True),
        flush=True,
    )

    print("Fitting final GAM on the complete training partition...", flush=True)
    gc.collect()
    final_fit_start = time.perf_counter()
    gam = build_gam()

    gam.fit(X_train.values, y_train.values)
    print(
        f"Final GAM fit completed in {time.perf_counter() - final_fit_start:.1f}s.",
        flush=True,
    )
    y_pred = gam.predict(X_test.values)
    print("Final test prediction completed.", flush=True)
    baseline_pred = np.full(len(y_test), y_train.mean())

    metrics = {
        "validation_method": "5-fold cross-validation on the 80% training partition",
        "test_split": 0.2,
        "random_state": RANDOM_STATE,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv_folds": CV_FOLDS,
        **cv_metrics,
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_r2": float(r2_score(y_test, baseline_pred)),
        "feature_order": FEATURE_ORDER,
    }

    # Build all new artifacts before replacing existing outputs. If the process
    # is killed by WSL, the previous valid metrics/effects remain untouched.
    effects_temporary_path = save_gam_effects(gam, X_train)
    gender_temporary_path = save_gender_predictions(gam, X_train)
    metrics_temporary_path = OUTPUT_DIR / "gam_metrics.json.tmp"
    with open(metrics_temporary_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    os.replace(metrics_temporary_path, OUTPUT_DIR / "gam_metrics.json")
    os.replace(effects_temporary_path, OUTPUT_DIR / "gam_effects.csv")
    os.replace(gender_temporary_path, OUTPUT_DIR / "gam_gender_predictions.csv")
    return metrics


def main():
    model_df = load_and_prepare_data()
    metrics = train_and_evaluate_gam(model_df)
    print("GAM evaluation results:")
    print(json.dumps(metrics, indent=2))
    print("Saved: outputs/gam_metrics.json and outputs/gam_effects.csv")


if __name__ == "__main__":
    main()
