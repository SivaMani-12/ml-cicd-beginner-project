
from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import precision_recall_curve

from src.data import FEATURES, TARGET, make_dataset, validate


# Minimum performance required to pass the CI/CD quality gate
MIN_F1 = 0.60

RANDOM_STATE = 42
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.20


def build_preprocessor():
    """Prepare numerical and categorical columns."""

    nums = [
        "tenure_months",
        "monthly_charges",
        "support_tickets",
    ]

    cats = [
        "contract_type",
        "uses_auto_pay",
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, nums),
        ("cat", categorical_pipeline, cats),
    ])


def build_models():
    """Create candidate models for comparison."""

    preprocessor = build_preprocessor()

    return {
        "logistic_regression": Pipeline([
            ("prep", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]),

        "random_forest": Pipeline([
            ("prep", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        "extra_trees": Pipeline([
            ("prep", clone(preprocessor)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def find_best_threshold(y_true, probabilities):
    """
    Select a classification threshold using validation data.

    The default threshold is 0.50, but another threshold
    may improve the balance between precision and recall.
    """

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.50, 0.0

    denominator = precisions[:-1] + recalls[:-1]

    f1_values = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    # Prefer a threshold closer to 0.50 if F1 scores tie.
    best_f1 = float(np.max(f1_values))
    tied_indices = np.flatnonzero(
        np.isclose(f1_values, best_f1)
    )

    best_index = min(
        tied_indices,
        key=lambda i: abs(float(thresholds[i]) - 0.50),
    )

    return (
        float(thresholds[best_index]),
        best_f1,
    )


def evaluate_model(model, X, y, threshold):
    """Calculate evaluation metrics."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": round(
            float(accuracy_score(y, predictions)), 4
        ),
        "precision": round(
            float(precision_score(
                y, predictions, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y, predictions, zero_division=0
            )), 4
        ),
        "f1": round(
            float(f1_score(
                y, predictions, zero_division=0
            )), 4
        ),
        "roc_auc": round(
            float(roc_auc_score(y, probabilities)), 4
        ),
        "test_rows": int(len(y)),
    }


def train(output="artifacts"):
    """Train, select, evaluate and save the best model."""

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate the dataset.
    df = make_dataset()
    validate(df)

    X = df[FEATURES]
    y = df[TARGET]

    # 2. Keep the test dataset separate.
    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # 3. Create training and validation datasets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    # 4. Train candidate models and compare validation F1.
    candidates = build_models()
    best_name = None
    best_model = None
    best_threshold = 0.50
    best_validation_f1 = -1.0

    validation_results = {}

    for name, model in candidates.items():
        print(f"\nTraining candidate: {name}")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        threshold, validation_f1 = find_best_threshold(
            y_val,
            probabilities,
        )

        validation_results[name] = {
            "validation_f1": round(validation_f1, 4),
            "threshold": round(threshold, 4),
        }

        print(
            f"Validation F1: {validation_f1:.4f} | "
            f"Threshold: {threshold:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_name = name
            best_model = model
            best_threshold = threshold

    # 5. Retrain the selected model on all non-test data.
    # The test data remains untouched until final evaluation.
    best_model.fit(X_train_full, y_train_full)

    # 6. Evaluate the final model on the held-out test set.
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        best_threshold,
    )

    metrics["min_f1_required"] = MIN_F1
    metrics["validation_f1"] = round(
        best_validation_f1, 4
    )
    metrics["decision_threshold"] = round(
        best_threshold, 4
    )

    quality_gate_passed = metrics["f1"] >= MIN_F1

    report = {
        "dataset": "synthetic demo data",
        "features": FEATURES,
        "target": TARGET,
        "selected_model": best_name,
        "candidate_validation_results": validation_results,
        "metrics": metrics,
        "quality_gate_passed": quality_gate_passed,
    }

    # 7. Save the model and reports.
    # model.joblib remains a regular sklearn Pipeline.
    joblib.dump(best_model, out / "model.joblib")

    # Bundle includes the threshold needed for predictions.
    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "model_name": best_name,
        },
        out / "model_bundle.joblib",
    )

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    (out / "evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL EVALUATION REPORT")
    print(json.dumps(report, indent=2))

    # 8. Enforce the CI/CD quality gate.
    if not quality_gate_passed:
        raise RuntimeError(
            f"Quality gate failed: "
            f"F1 {metrics['f1']} < {MIN_F1}"
        )

    print("\nQUALITY GATE PASSED")
    print("The model meets the minimum F1 requirement.")

    return report
