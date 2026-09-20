from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    return model_df.dropna().copy()


def make_baseline_row(reference_df: pd.DataFrame) -> np.ndarray:
    values = []
    for feature in FEATURE_ORDER:
        if feature in CATEGORICAL_FEATURES:
            values.append(float(reference_df[feature].mode().iloc[0]))
        else:
            values.append(float(reference_df[feature].median()))
    return np.asarray(values, dtype=float)


def save_gam_effects(gam: LinearGAM, reference_df: pd.DataFrame) -> None:
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

    pd.DataFrame(effect_rows).to_csv(OUTPUT_DIR / "gam_effects.csv", index=False)


def train_and_evaluate_gam(model_df: pd.DataFrame) -> dict:
    X = model_df[FEATURE_ORDER]
    y = model_df["earnings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training GAM on {len(X_train):,} rows; testing on {len(X_test):,} rows...")

    gam = LinearGAM(
        s(0, n_splines=18)
        + s(1, n_splines=18)
        + f(2)
        + f(3)
        + f(4)
        + f(5)
    )

    gam.fit(X_train.values, y_train.values)
    y_pred = gam.predict(X_test.values)

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "feature_order": FEATURE_ORDER,
    }

    with open(OUTPUT_DIR / "gam_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    save_gam_effects(gam, X_train)
    return metrics


def main():
    model_df = load_and_prepare_data()
    metrics = train_and_evaluate_gam(model_df)
    print("GAM evaluation results:")
    print(json.dumps(metrics, indent=2))
    print("Saved: outputs/gam_metrics.json and outputs/gam_effects.csv")


if __name__ == "__main__":
    main()
