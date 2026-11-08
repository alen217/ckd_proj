"""Inspect dataset and verify the saved CKD model's required pipeline structure."""

import joblib
import pandas as pd

from ckd_preprocessing import CATEGORY_MAPPINGS


DATASET_PATH = "kidney_disease.csv"
MODEL_PATH = "kidney_model.joblib"
EXPECTED_FEATURES = 24
EXPECTED_ENGINEERED_FEATURES = {
    "anemia_risk",
    "hypertension_risk",
    "diabetes_risk",
    "combined_risk_index",
    "kidney_function_score",
    "bp_severity",
    "hemoglobin_status",
}


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = (
        df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    )
    target = (
        df["classification"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z]", "", regex=True)
    )
    target_counts = target.value_counts()
    feature_columns = [
        column
        for column in df.columns
        if column not in {"id", "classification"}
    ]

    print(f"Dataset shape (including ID and target): {df.shape}")
    print(f"Model input feature count: {len(feature_columns)}")
    print("Target distribution:")
    print(target_counts)
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    steps = list(model.named_steps)
    required_order = ["preprocessor", "scaler", "smote", "classifier"]
    pipeline_valid = steps == required_order
    names = set(model.named_steps["preprocessor"].get_feature_names_out())
    engineered_valid = EXPECTED_ENGINEERED_FEATURES <= names
    mappings_valid = artifact.get("category_mappings") == CATEGORY_MAPPINGS
    features_valid = len(artifact.get("feature_names", [])) == EXPECTED_FEATURES

    print("\nSaved model:", type(model.named_steps["classifier"]).__name__)
    print("Pipeline steps:", " -> ".join(steps))
    print("Input feature count:", len(artifact.get("feature_names", [])))
    print("Scaler before SMOTE and classifier:", pipeline_valid)
    print("Seven engineered features present:", engineered_valid)
    print("Shared categorical maps saved:", mappings_valid)
    print("Training metrics:", artifact.get("training_metrics", "not present"))
    print("Test metrics:", artifact.get("metrics", "not present"))
    print("Five-fold CV ROC-AUC:", artifact.get("cv_roc_auc", "not present"))

    structural_checks = [
        len(df) == 400,
        len(feature_columns) == EXPECTED_FEATURES,
        pipeline_valid,
        engineered_valid,
        mappings_valid,
        features_valid,
    ]
    print("\nStructural requirements:", "PASS" if all(structural_checks) else "FAIL")
    if not all(structural_checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
