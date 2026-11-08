"""Train, evaluate, and report leak-free CKD classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.shared import Inches
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ckd_preprocessing import (
    CATEGORY_MAPPINGS,
    CKDPreprocessor,
    categorical_columns,
    numeric_columns,
)


RANDOM_STATE = 42
DATASET_PATH = Path("kidney_disease.csv")
OUTPUT_DIR = Path("reports")
FIGURE_DIR = OUTPUT_DIR / "figures"
ENGINEERED_FEATURES = [
    "anemia_risk",
    "hypertension_risk",
    "diabetes_risk",
    "combined_risk_index",
    "kidney_function_score",
    "bp_severity",
    "hemoglobin_status",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw clinical inputs and encode the outcome without fitting imputers."""
    df = pd.read_csv(DATASET_PATH)
    df.columns = (
        df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    )
    df = df.drop(columns=["id"], errors="ignore")

    required = numeric_columns + categorical_columns + ["classification"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[required].copy()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"?": pd.NA, "": pd.NA})
        )

    target = (
        df.pop("classification")
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z]", "", regex=True)
        .map({"ckd": 1, "notckd": 0})
    )
    if target.isna().any():
        raise ValueError("Target contains missing or unrecognized labels.")
    if len(df) != 400 or len(df.columns) != 24:
        raise ValueError(
            f"Expected 400 records and 24 input features; got "
            f"{len(df)} records and {len(df.columns)} features."
        )
    return df, target.astype(int)


def make_pipeline(classifier) -> Pipeline:
    """Keep learned cleaning, scaling, and SMOTE inside every CV training fold."""
    return Pipeline(
        [
            ("preprocessor", CKDPreprocessor()),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def make_eda_frame(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Create visualization-only imputation; these values never enter model fitting."""
    display = CKDPreprocessor().fit(X).transform(X)
    display["classification"] = y.to_numpy()
    return display


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_eda_figures(eda: pd.DataFrame) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    labels = eda["classification"].map({0: "Not CKD", 1: "CKD"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = labels.value_counts().reindex(["Not CKD", "CKD"])
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    sns.barplot(x=counts.index, y=counts.values, ax=axes[1], color="#4c9f70")
    axes[1].set(title="Target counts", xlabel="Class", ylabel="Records")
    fig.suptitle("Target distribution")
    paths["Target distribution (pie and bar)"] = save_figure(
        fig, "target_distribution.png"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feature in zip(axes.flat, ["age", "bp", "hemo", "sc"]):
        sns.histplot(
            data=eda.assign(classification=labels),
            x=feature,
            hue="classification",
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"{feature} by CKD status")
    paths["Clinical feature distributions"] = save_figure(
        fig, "clinical_distributions.png"
    )

    numeric_eda = eda.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(17, 14))
    sns.heatmap(numeric_eda.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation heatmap of numeric and engineered features")
    paths["Correlation heatmap"] = save_figure(fig, "correlation_heatmap.png")

    correlations = (
        numeric_eda.corr()["classification"]
        .drop("classification")
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    correlations.plot(kind="barh", ax=ax, color="#4c9f70")
    ax.set_title("Feature correlation with target")
    ax.set_xlabel("Pearson correlation")
    paths["Feature-target correlations"] = save_figure(
        fig, "feature_target_correlations.png"
    )
    return paths


def evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return metrics, predictions, probabilities


def model_feature_importance(model: Pipeline) -> pd.Series:
    classifier = model.named_steps["classifier"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        raise TypeError(f"Unsupported classifier for feature importance: {type(classifier)}")
    return pd.Series(values, index=names).sort_values(ascending=False)


def generate_evaluation_figures(
    models: dict[str, Pipeline],
    metrics: dict[str, dict[str, float]],
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    y_test: pd.Series,
    selected_name: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, probability in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_test, probability)
        score = auc(false_positive, true_positive)
        ax.plot(false_positive, true_positive, label=f"{name} (AUC={score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")
    paths["ROC curves"] = save_figure(fig, "roc_curves.png")

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, prediction) in zip(axes, predictions.items()):
        sns.heatmap(
            confusion_matrix(y_test, prediction, labels=[0, 1]),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not CKD", "CKD"],
            yticklabels=["Not CKD", "CKD"],
            ax=ax,
        )
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    paths["Confusion matrices"] = save_figure(fig, "confusion_matrices.png")

    importance = model_feature_importance(models[selected_name]).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    importance.plot(kind="barh", ax=ax, color="#4c9f70")
    ax.set(title=f"Top 15 features ({selected_name})", xlabel="Importance")
    paths["Top 15 feature importance"] = save_figure(
        fig, "feature_importance_top15.png"
    )

    metric_frame = pd.DataFrame(metrics).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    metric_frame.plot(kind="bar", ax=ax)
    ax.set(ylim=(0, 1.05), ylabel="Score", title="Test-set model comparison")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    paths["Model performance comparison"] = save_figure(
        fig, "model_performance.png"
    )
    return paths


def add_report_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)


def generate_report(
    X: pd.DataFrame,
    y: pd.Series,
    cv_scores: dict[str, np.ndarray],
    training_metrics: dict[str, dict[str, float]],
    metrics: dict[str, dict[str, float]],
    reports: dict[str, str],
    models: dict[str, Pipeline],
    figures: dict[str, Path],
    selected_name: str,
) -> Path:
    document = Document()
    document.add_heading("Chronic Kidney Disease Prediction", 0)
    document.add_paragraph(
        "Model development and evaluation report. This system is for research "
        "and screening support only; it is not a diagnosis or a substitute for "
        "clinical assessment."
    )
    document.add_heading("Executive summary", level=1)
    document.add_paragraph(
        f"The source dataset contains {len(X)} records and {X.shape[1]} clinical "
        f"input features. The stratified holdout contains {int(len(y) * 0.2)} "
        "patients. The selected model, based on training-set cross-validation "
        f"ROC-AUC, is {selected_name}. No requested performance threshold is "
        "assumed; the reported scores are measured on the held-out test set."
    )

    document.add_heading("Dataset and target", level=1)
    add_report_table(
        document,
        ["Measure", "Value"],
        [
            ["Records", str(len(X))],
            ["Clinical input features", str(X.shape[1])],
            ["CKD (1)", str(int(y.sum()))],
            ["Not CKD (0)", str(int((y == 0).sum()))],
            ["Split", "80/20, random_state=42, stratified"],
        ],
    )
    document.add_paragraph(
        "The ID field is discarded. Column names are stripped, lowercased, and "
        "normalized. Numeric columns are coerced to numeric values; unknown "
        "categorical labels and missing targets are rejected."
    )

    document.add_heading("Missing values, mappings, and engineered features", level=1)
    document.add_paragraph(
        "Numeric medians and categorical modes are learned from training data "
        "(and refit within each cross-validation fold). The shared categorical "
        "mapping uses normal/good/no/notpresent = 0 and abnormal/poor/yes/present = 1."
    )
    add_report_table(
        document,
        ["Feature", "Definition"],
        [
            ["anemia_risk", "1 when hemo < 12"],
            ["hypertension_risk", "1 when bp > 140"],
            ["diabetes_risk", "1 when dm = 1"],
            ["combined_risk_index", "Sum of anemia, hypertension, and diabetes risk"],
            ["kidney_function_score", "bu / max(sc, 0.1)"],
            ["bp_severity", "<120:0, <130:1, <140:2, <180:3, otherwise:4"],
            ["hemoglobin_status", "<8:0, <10:1, <12:2, <16:3, otherwise:4"],
        ],
    )

    document.add_heading("Leak-free modeling protocol", level=1)
    document.add_paragraph(
        "The 80/20 stratified split is made before any resampling. Every classifier "
        "uses an imbalanced-learn pipeline: fold-fitted CKD preprocessing, "
        "StandardScaler, SMOTE(random_state=42), then classifier. Cross-validation "
        "uses StratifiedKFold(n_splits=5, shuffle=True, random_state=42) and "
        "ROC-AUC scoring. Thus SMOTE is fitted only on each training fold."
    )
    document.add_paragraph(
        "Random Forest uses 300 trees, depth 12, minimum split 5, minimum leaf 3, "
        "sqrt feature sampling, and balanced class weights. Logistic Regression "
        "uses C=0.5, L2, lbfgs, and max_iter=2000. XGBoost parameters are selected "
        "by randomized search using the same training-only five-fold procedure."
    )

    document.add_heading("Cross-validation results", level=1)
    add_report_table(
        document,
        ["Model", "Mean ROC-AUC", "Std. dev."],
        [
            [
                name,
                f"{scores.mean():.4f}",
                f"{scores.std():.4f}",
            ]
            for name, scores in cv_scores.items()
        ],
    )
    document.add_heading("Training-set results", level=1)
    add_report_table(
        document,
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
            [
                name,
                *[f"{value:.4f}" for value in values.values()],
            ]
            for name, values in training_metrics.items()
        ],
    )
    document.add_heading("Held-out test results", level=1)
    add_report_table(
        document,
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
            [
                name,
                *[f"{value:.4f}" for value in values.values()],
            ]
            for name, values in metrics.items()
        ],
    )
    for name, report_text in reports.items():
        document.add_heading(f"{name} classification report", level=2)
        document.add_paragraph(report_text)

    document.add_heading("Exploratory analysis and evaluation figures", level=1)
    for title, image_path in figures.items():
        document.add_page_break()
        document.add_heading(title, level=2)
        document.add_picture(str(image_path), width=Inches(6.2))
        document.add_paragraph(
            "Figure generated from the supplied dataset or held-out test set as "
            "identified in the title; charts are embedded in this document."
        )

    importance = model_feature_importance(models[selected_name])
    document.add_heading("Feature importance and limitations", level=1)
    document.add_paragraph(
        "Importance is model-specific (tree importance or absolute logistic "
        "coefficient) and is not causal evidence. Serum creatinine is retained. "
        "The engineered kidney_function_score and combined_risk_index are "
        "included and their observed ranks are listed below."
    )
    add_report_table(
        document,
        ["Feature", "Importance", "Rank"],
        [
            [
                feature,
                f"{importance.get(feature, 0.0):.6f}",
                str(int(importance.index.get_loc(feature)) + 1)
                if feature in importance.index
                else "not available",
            ]
            for feature in ["sc", "hemo", "pcv", "bu", "kidney_function_score", "combined_risk_index"]
        ],
    )
    document.add_paragraph(
        "Model scores on a small public dataset do not establish clinical "
        "validity, calibration, generalizability, or suitability for patient "
        "care. Independent external validation and clinician oversight are "
        "required. A requested threshold is never guaranteed or used to select "
        "test-set outcomes."
    )

    document.add_heading("Appendix: per-feature data profiles", level=1)
    for index, feature in enumerate(numeric_columns + categorical_columns):
        if index:
            document.add_page_break()
        document.add_heading(f"Feature profile: {feature}", level=2)
        document.add_paragraph(
            "Descriptive statistics are calculated on the source dataset before "
            "model fitting. They are for documentation only; model imputers are "
            "fitted only on training data or the corresponding CV training fold."
        )
        if feature in numeric_columns:
            values = pd.to_numeric(X[feature], errors="coerce")
            summary = values.describe()
            add_report_table(
                document,
                ["Statistic", "Value"],
                [
                    ["Missing values", str(int(values.isna().sum()))],
                    *[
                        [statistic, f"{summary[statistic]:.4f}"]
                        for statistic in ["mean", "std", "min", "25%", "50%", "75%", "max"]
                    ],
                    [
                        "Mean in Not CKD",
                        f"{values[y == 0].mean():.4f}",
                    ],
                    [
                        "Mean in CKD",
                        f"{values[y == 1].mean():.4f}",
                    ],
                ],
            )
        else:
            values = (
                X[feature]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"?": pd.NA, "": pd.NA})
                .fillna("(missing)")
            )
            category_rows = [
                [str(value), str(int(count))]
                for value, count in values.value_counts(dropna=False).items()
            ]
            add_report_table(
                document,
                ["Category", "Record count"],
                category_rows,
            )
        document.add_paragraph(
            "Interpretation must account for missingness, measurement practices, "
            "and the dataset's limited size; this profile is not a clinical "
            "reference range."
        )

    document.add_heading("Dataset link", level=1)
    document.add_paragraph(
        "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease"
    )
    path = OUTPUT_DIR / "CKD_Project_Report.docx"
    document.save(path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    print(f"Dataset shape: ({len(X)}, {X.shape[1]})")
    print("Target distribution (0=Not CKD, 1=CKD):")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train/test sizes: {len(X_train)}/{len(X_test)}")
    print("Train ratio:", y_train.value_counts(normalize=True).sort_index().to_dict())
    print("Test ratio:", y_test.value_counts(normalize=True).sort_index().to_dict())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    )
    lr = make_pipeline(
        LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    )
    xgb = make_pipeline(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )
    )
    xgb_parameter_space = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
        }
    xgb_nested_search = RandomizedSearchCV(
        xgb,
        xgb_parameter_space,
        n_iter=4,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    print("Estimating XGBoost ROC-AUC with nested five-fold CV...")
    xgb_cv_scores = cross_val_score(
        xgb_nested_search,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )
    xgb_search = RandomizedSearchCV(
        xgb,
        xgb_parameter_space,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    print("Tuning XGBoost within training folds...")
    xgb_search.fit(X_train, y_train)

    models = {
        "Random Forest": rf,
        "XGBoost": xgb_search.best_estimator_,
        "Logistic Regression": lr,
    }
    print("XGBoost best parameters:", xgb_search.best_params_)

    cv_scores: dict[str, np.ndarray] = {"XGBoost": xgb_cv_scores}
    training_metrics: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    report_text: dict[str, str] = {}
    for name, model in models.items():
        if name != "XGBoost":
            print(f"Running five-fold ROC-AUC cross-validation: {name}")
            cv_scores[name] = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,
                error_score="raise",
            )
        model.fit(X_train, y_train)
        training_metrics[name], _, _ = evaluate(model, X_train, y_train)
        metrics[name], predictions[name], probabilities[name] = evaluate(
            model, X_test, y_test
        )
        report_text[name] = classification_report(
            y_test,
            predictions[name],
            labels=[0, 1],
            target_names=["Not CKD", "CKD"],
            zero_division=0,
        )
        print(
            f"{name}: training metrics {training_metrics[name]}; "
            f"nested CV ROC-AUC {cv_scores[name].mean():.4f} "
            f"+/- {cv_scores[name].std():.4f}; test metrics {metrics[name]}"
        )
        print("Confusion matrix:\n", confusion_matrix(y_test, predictions[name]))
        print(report_text[name])

    selected_name = max(cv_scores, key=lambda name: cv_scores[name].mean())
    print("Selected model by mean training CV ROC-AUC:", selected_name)
    selected_metrics = metrics[selected_name]
    print(
        "Threshold check (measured, not guaranteed):",
        {
            "accuracy_at_least_0.95": selected_metrics["accuracy"] >= 0.95,
            "roc_auc_at_least_0.98": selected_metrics["roc_auc"] >= 0.98,
            "recall_at_least_0.95": selected_metrics["recall"] >= 0.95,
        },
    )

    eda = make_eda_frame(X, y)
    figures = generate_eda_figures(eda)
    figures.update(
        generate_evaluation_figures(
            models,
            metrics,
            predictions,
            probabilities,
            y_test,
            selected_name,
        )
    )
    importance = model_feature_importance(models[selected_name])
    print("Top 15 features:\n", importance.head(15))
    print(
        "Engineered features in top 10:",
        [feature for feature in ENGINEERED_FEATURES if feature in importance.head(10)],
    )

    artifact = {
        "model": models[selected_name],
        "models": models,
        "feature_names": list(X.columns),
        "selected_model": selected_name,
        "metrics": metrics,
        "training_metrics": training_metrics,
        "cv_roc_auc": {
            name: {"mean": float(scores.mean()), "std": float(scores.std())}
            for name, scores in cv_scores.items()
        },
        "category_mappings": CATEGORY_MAPPINGS,
        "engineered_features": ENGINEERED_FEATURES,
    }
    joblib.dump(artifact, "kidney_model.joblib")
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "test_metrics": metrics,
                "training_metrics": training_metrics,
                "cv_roc_auc": artifact["cv_roc_auc"],
                "xgboost_best_params": xgb_search.best_params_,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved model: kidney_model.joblib")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
