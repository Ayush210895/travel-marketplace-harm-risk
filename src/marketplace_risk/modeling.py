"""Train and evaluate capacity-constrained booking-risk models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from marketplace_risk.preprocessing import (
    AdrOutlierCapper,
    BookingValueFeatures,
    GuestCountFeatures,
    MissingSignalFeatures,
    NOMINAL_COLUMNS,
    RareCategoryCleaner,
    TemporalFeatures,
)
from marketplace_risk.schema import TARGET_COLUMN, model_feature_columns


def capacity_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    harm_amount: np.ndarray | None = None,
    review_fraction: float = 0.05,
    probability_scores: bool = True,
) -> dict[str, float | int]:
    """Evaluate ranking quality when only a fixed fraction can be reviewed."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("labels and scores must have the same non-zero length")
    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")

    review_count = max(1, math.ceil(len(labels) * review_fraction))
    ranked = np.argsort(-scores, kind="stable")
    reviewed = ranked[:review_count]
    caught = int(labels[reviewed].sum())
    positives = int(labels.sum())
    metrics: dict[str, float | int] = {
        "rows": len(labels),
        "harmful_bookings": positives,
        "review_count": review_count,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision_at_5pct": caught / review_count,
        "recall_at_5pct": caught / positives if positives else 0.0,
    }
    if probability_scores:
        metrics["brier_score"] = float(brier_score_loss(labels, scores))
    if harm_amount is not None:
        amount = np.asarray(harm_amount, dtype=float)
        total_harm = float(amount[labels == 1].sum())
        captured_harm = float(amount[reviewed][labels[reviewed] == 1].sum())
        metrics["harm_value_capture_at_5pct"] = (
            captured_harm / total_harm if total_harm else 0.0
        )
    return metrics


def recall_at_capacity(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(capacity_metrics(labels, scores)["recall_at_5pct"])


def rolling_mature_splits(
    data: pd.DataFrame,
    validation_fractions: tuple[float, ...] = (0.55, 0.70, 0.85),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding temporal folds with mature labels at each train cutoff."""
    ordered = data.sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    row_count = len(ordered)
    boundaries = [int(row_count * fraction) for fraction in validation_fractions]
    boundaries.append(row_count)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        cutoff = ordered.loc[start, "booking_timestamp"]
        train_mask = (
            (ordered["booking_timestamp"] < cutoff)
            & (ordered["outcome_matured_at"] <= cutoff)
        )
        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.arange(start, end)
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("Rolling fold contains an empty train or validation period")
        splits.append((train_indices, validation_indices))
    return splits


def _feature_engineering() -> Pipeline:
    return Pipeline(
        [
            ("categories", RareCategoryCleaner(min_frequency=50)),
            ("adr", AdrOutlierCapper(quantile=0.999)),
            ("guests", GuestCountFeatures()),
            ("time", TemporalFeatures()),
            ("booking_value", BookingValueFeatures()),
            ("missing_signals", MissingSignalFeatures()),
        ]
    )


def _engineered_columns(sample: pd.DataFrame) -> tuple[list[str], list[str]]:
    engineered = _feature_engineering().fit_transform(sample)
    categorical = [column for column in NOMINAL_COLUMNS if column in engineered]
    numeric = [
        column
        for column in engineered.columns
        if column not in categorical and pd.api.types.is_numeric_dtype(engineered[column])
    ]
    return categorical, numeric


def build_logistic_pipeline(
    sample: pd.DataFrame,
    class_weight: str | None = None,
) -> Pipeline:
    categorical, numeric = _engineered_columns(sample)
    numeric = [column for column in numeric if column != "estimated_booking_value"]
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", _feature_engineering()),
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_xgboost_pipeline(sample: pd.DataFrame) -> Pipeline:
    categorical, numeric = _engineered_columns(sample)
    numeric = [column for column in numeric if column != "log_estimated_booking_value"]
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    return Pipeline(
        [
            ("features", _feature_engineering()),
            ("preprocess", preprocessing),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def train_models(
    data_path: Path,
    manifest_path: Path,
    output_dir: Path,
    search_iterations: int = 12,
) -> dict[str, object]:
    data = pd.read_csv(
        data_path, parse_dates=["booking_timestamp", "outcome_matured_at"]
    ).sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_start = pd.Timestamp(manifest["validation_start"])
    test_start = pd.Timestamp(manifest["test_start"])

    development = data[data["booking_timestamp"] < validation_start].copy()
    train = development[development["outcome_matured_at"] <= validation_start].copy()
    validation = data[
        (data["booking_timestamp"] >= validation_start)
        & (data["booking_timestamp"] < test_start)
    ].copy()

    feature_columns = model_feature_columns(data.columns.tolist())
    train_x, train_y = train[feature_columns], train[TARGET_COLUMN].to_numpy()
    validation_x = validation[feature_columns]
    validation_y = validation[TARGET_COLUMN].to_numpy()

    logistic = build_logistic_pipeline(train_x, class_weight=None)
    logistic.fit(train_x, train_y)
    logistic_scores = logistic.predict_proba(validation_x)[:, 1]
    logistic_engineered = logistic.named_steps["features"].transform(validation_x)
    logistic_expected_loss = (
        logistic_scores * logistic_engineered["estimated_booking_value"].to_numpy()
    )

    weighted_logistic = build_logistic_pipeline(train_x, class_weight="balanced")
    weighted_logistic.fit(train_x, train_y)
    weighted_logistic_scores = weighted_logistic.predict_proba(validation_x)[:, 1]

    xgboost = build_xgboost_pipeline(development[feature_columns])
    folds = rolling_mature_splits(development)
    negative_to_positive = float((train_y == 0).sum() / (train_y == 1).sum())
    parameter_space = {
        "model__n_estimators": randint(200, 701),
        "model__learning_rate": loguniform(0.02, 0.20),
        "model__max_depth": randint(3, 8),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.70, 0.30),
        "model__colsample_bytree": uniform(0.60, 0.40),
        "model__gamma": uniform(0.0, 1.0),
        "model__reg_alpha": loguniform(1e-3, 2.0),
        "model__reg_lambda": loguniform(0.5, 20.0),
        "model__scale_pos_weight": [
            1.0,
            negative_to_positive * 0.75,
            negative_to_positive,
            negative_to_positive * 1.25,
        ],
    }

    def scorer(estimator: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> float:
        return recall_at_capacity(labels, estimator.predict_proba(features)[:, 1])

    search = RandomizedSearchCV(
        estimator=xgboost,
        param_distributions=parameter_space,
        n_iter=search_iterations,
        scoring=scorer,
        cv=folds,
        refit=False,
        random_state=42,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    development_x = development[feature_columns]
    development_y = development[TARGET_COLUMN].to_numpy()
    search.fit(development_x, development_y)

    best_parameters = search.cv_results_["params"][search.best_index_]
    tuned_xgboost = build_xgboost_pipeline(train_x).set_params(**best_parameters)
    tuned_xgboost.fit(train_x, train_y)
    xgboost_scores = tuned_xgboost.predict_proba(validation_x)[:, 1]
    xgboost_engineered = tuned_xgboost.named_steps["features"].transform(validation_x)
    xgboost_expected_loss = (
        xgboost_scores * xgboost_engineered["estimated_booking_value"].to_numpy()
    )

    results: dict[str, object] = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "xgboost_search_iterations": search_iterations,
        "xgboost_best_cv_recall_at_5pct": float(search.best_score_),
        "xgboost_best_parameters": best_parameters,
        "validation": {
            "logistic_regression": capacity_metrics(
                validation_y,
                logistic_scores,
                validation["harm_amount"].to_numpy(),
            ),
            "logistic_regression_balanced": capacity_metrics(
                validation_y,
                weighted_logistic_scores,
                validation["harm_amount"].to_numpy(),
            ),
            "xgboost": capacity_metrics(
                validation_y,
                xgboost_scores,
                validation["harm_amount"].to_numpy(),
            ),
            "logistic_expected_loss_ranking": capacity_metrics(
                validation_y,
                logistic_expected_loss,
                validation["harm_amount"].to_numpy(),
                probability_scores=False,
            ),
            "xgboost_expected_loss_ranking": capacity_metrics(
                validation_y,
                xgboost_expected_loss,
                validation["harm_amount"].to_numpy(),
                probability_scores=False,
            ),
        },
        "current_operating_policy": {
            "model": "xgboost",
            "ranking_score": "predicted_harm_probability * estimated_booking_value",
            "review_fraction": 0.05,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, output_dir / "logistic_regression.joblib")
    joblib.dump(
        weighted_logistic, output_dir / "logistic_regression_balanced.joblib"
    )
    joblib.dump(tuned_xgboost, output_dir / "xgboost.joblib")
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(search.cv_results_).to_csv(
        output_dir / "xgboost_search_results.csv", index=False
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-iterations", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_models(
        data_path=args.data,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        search_iterations=args.search_iterations,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
