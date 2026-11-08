
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# =========================================================
# 1. FEATURE DEFINITIONS
# =========================================================

numeric_columns = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

categorical_columns = [
    "rbc", "pc", "pcc", "ba", "appet",
    "htn", "dm", "cad", "pe", "ane"
]


# =========================================================
# 2. CATEGORICAL MAPPINGS
# =========================================================

CATEGORY_MAPPINGS = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "appet": {"good": 0, "poor": 1},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0}
}


# =========================================================
# 3. CUSTOM CKD PREPROCESSOR
# =========================================================

class CKDPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.mappings = CATEGORY_MAPPINGS

    def fit(self, X, y=None):

        X = X.copy()

        # Numeric median imputation
        self.numeric_medians_ = X[
            self.numeric_columns
        ].median()

        self.numeric_medians_ = self.numeric_medians_.fillna(0)

        # Categorical mode imputation
        self.categorical_modes_ = {}

        for col in self.categorical_columns:

            values = (
                X[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace("?", pd.NA)
            )

            mode_values = values.mode(dropna=True)

            self.categorical_modes_[col] = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "unknown"
            )

        # Kidney function score median
        bu_values = pd.to_numeric(
            X["bu"], errors="coerce"
        ).fillna(self.numeric_medians_["bu"])

        sc_values = pd.to_numeric(
            X["sc"], errors="coerce"
        ).fillna(self.numeric_medians_["sc"])

        ratio = bu_values / sc_values.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        self.kidney_function_score_median_ = ratio.median()

        if pd.isna(self.kidney_function_score_median_):
            self.kidney_function_score_median_ = 0.0

        return self

    def transform(self, X):

        X = X.copy()

        # Numeric cleaning and imputation
        for col in self.numeric_columns:

            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(self.numeric_medians_[col])

        # Categorical cleaning and imputation
        for col in self.categorical_columns:

            X[col] = (
                X[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace("?", pd.NA)
            )

            X[col] = X[col].fillna(self.categorical_modes_[col])

        # Apply categorical mappings
        for col, mapping in self.mappings.items():

            original_values = X[col].copy()
            X[col] = X[col].map(mapping)

            if X[col].isnull().any():

                unknown_values = original_values[X[col].isnull()].unique()

                raise ValueError(
                    f"Unknown categorical value(s) in {col}: "
                    f"{unknown_values}"
                )

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        X["anemia_risk"] = (X["hemo"] < 12).astype(int)
        X["hypertension_risk"] = (X["bp"] > 140).astype(int)
        X["diabetes_risk"] = X["dm"]

        X["combined_risk_index"] = (
            X["anemia_risk"]
            + X["hypertension_risk"]
            + X["diabetes_risk"]
        )

        # BUN / Serum Creatinine ratio
        X["kidney_function_score"] = (
            X["bu"] / X["sc"].clip(lower=0.1)
        )

        X["kidney_function_score"] = (
            X["kidney_function_score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.kidney_function_score_median_)
        )

        # Ordinal bands: normal, elevated, stage 1, stage 2, crisis.
        X["bp_severity"] = np.select(
            [
                X["bp"] < 120,
                X["bp"] < 130,
                X["bp"] < 140,
                X["bp"] < 180,
            ],
            [0, 1, 2, 3],
            default=4
        )

        # Ordinal bands from severe anemia through elevated hemoglobin.
        X["hemoglobin_status"] = np.select(
            [
                X["hemo"] < 8,
                X["hemo"] < 10,
                X["hemo"] < 12,
                X["hemo"] < 16,
            ],
            [0, 1, 2, 3],
            default=4
        )

        return X.astype(float)

    def get_feature_names_out(self, input_features=None):

        return np.array(
            list(self.numeric_columns)
            + list(self.categorical_columns)
            + [
                "anemia_risk",
                "hypertension_risk",
                "diabetes_risk",
                "combined_risk_index",
                "kidney_function_score",
                "bp_severity",
                "hemoglobin_status"
            ],
            dtype=object
        )
