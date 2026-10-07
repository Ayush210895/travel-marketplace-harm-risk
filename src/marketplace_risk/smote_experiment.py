"""Compare SMOTENC sampling levels on the untouched development validation period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from marketplace_risk.modeling import _engineered_columns, _feature_engineering, capacity_metrics
from marketplace_risk.schema import TARGET_COLUMN, model_feature_columns


FEATURE_STEP_NAMES = [
    "categories",
    "adr",
    "guests",
    "time",
    "booking_value",
    "missing_signals",
]


def build_smotenc_logistic_pipeline(
    sample: pd.DataFrame,
    sampling_strategy: float,
) -> Pipeline:
    categorical, numeric = _engineered_columns(sample)
    numeric = [column for column in numeric if column != "estimated_booking_value"]
    categorical_indices = list(range(len(categorical)))
    numeric_indices = list(range(len(categorical), len(categorical) + len(numeric)))

    before_smote = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical,
            ),
            (
                "numeric",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        sparse_threshold=0,
    )
    after_smote = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_indices,
            ),
            ("numeric", "passthrough", numeric_indices),
        ]
    )
    return Pipeline(
        [
            *_feature_engineering().steps,
            ("before_smote", before_smote),
            (
                "smote",
                SMOTENC(
                    categorical_features=categorical_indices,
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=5,
                ),
            ),
            ("after_smote", after_smote),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def run_experiment(
    data_path: Path,
    manifest_path: Path,
    output_path: Path,
    strategies: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, object]:
    data = pd.read_csv(
        data_path, parse_dates=["booking_timestamp", "outcome_matured_at"]
    ).sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_start = pd.Timestamp(manifest["validation_start"])
    test_start = pd.Timestamp(manifest["test_start"])
    train = data[
        (data["booking_timestamp"] < validation_start)
        & (data["outcome_matured_at"] <= validation_start)
    ].copy()
    validation = data[
        (data["booking_timestamp"] >= validation_start)
        & (data["booking_timestamp"] < test_start)
    ].copy()
    columns = model_feature_columns(data.columns.tolist())
    train_x, train_y = train[columns], train[TARGET_COLUMN].to_numpy()
    validation_x, validation_y = (
        validation[columns],
        validation[TARGET_COLUMN].to_numpy(),
    )

    results: dict[str, object] = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "natural_training_harm_rate": float(train_y.mean()),
        "strategies": {},
    }
    best_pipeline: Pipeline | None = None
    best_value_capture = -np.inf
    best_strategy = None
    for strategy in strategies:
        pipeline = build_smotenc_logistic_pipeline(train_x, strategy)
        pipeline.fit(train_x, train_y)
        probability = pipeline.predict_proba(validation_x)[:, 1]
        engineered = validation_x.copy()
        for step_name in FEATURE_STEP_NAMES:
            engineered = pipeline.named_steps[step_name].transform(engineered)
        expected_loss = probability * engineered["estimated_booking_value"].to_numpy()
        metrics = capacity_metrics(
            validation_y,
            expected_loss,
            validation["harm_amount"].to_numpy(),
            probability_scores=False,
        )
        results["strategies"][str(strategy)] = metrics
        value_capture = float(metrics["harm_value_capture_at_5pct"])
        if value_capture > best_value_capture:
            best_value_capture = value_capture
            best_pipeline = pipeline
            best_strategy = strategy

    results["best_strategy"] = best_strategy
    results["best_harm_value_capture_at_5pct"] = best_value_capture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if best_pipeline is not None:
        joblib.dump(best_pipeline, output_path.with_suffix(".joblib"))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args.data, args.manifest, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
