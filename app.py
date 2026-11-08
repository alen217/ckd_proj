
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
saved_data = joblib.load("kidney_model.joblib")

model = saved_data["model"]
feature_names = saved_data["feature_names"]

# Numeric features
numeric_features = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]


def top_patient_feature_sensitivities(patient_data, limit=5):
    """Return one-at-a-time sensitivity, not causal explanations."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    ckd_index = list(classifier.classes_).index(1)
    patient_probability = float(model.predict_proba(patient_data)[0][ckd_index])
    baseline = {
        **preprocessor.numeric_medians_.to_dict(),
        **preprocessor.categorical_modes_,
    }

    sensitivities = []
    for feature, baseline_value in baseline.items():
        baseline_patient = patient_data.copy()
        baseline_patient.loc[:, feature] = baseline_value
        baseline_probability = float(
            model.predict_proba(baseline_patient)[0][ckd_index]
        )
        sensitivities.append({
            "feature": feature,
            "probability_change_percentage_points": round(
                (patient_probability - baseline_probability) * 100, 2
            ),
            "input_value": str(patient_data.iloc[0][feature]),
        })

    return sorted(
        sensitivities,
        key=lambda item: abs(item["probability_change_percentage_points"]),
        reverse=True,
    )[:limit]


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=saved_data.get("selected_model", "CKD classifier"),
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON object of patient features is required"}), 400

        # Check missing features
        missing = [f for f in feature_names if f not in data]

        if missing:
            return jsonify({
                "error": "Missing features",
                "missing_features": missing
            }), 400

        # Convert input values
        patient_data = {}

        for feature in feature_names:
            value = data[feature]

            if feature in numeric_features:
                patient_data[feature] = float(value)
            else:
                patient_data[feature] = str(value).strip()

        # Create DataFrame
        input_df = pd.DataFrame(
            [patient_data],
            columns=feature_names
        )

        # Prediction
        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        ckd_index = classes.index(1)

        ckd_probability = float(probabilities[ckd_index])
        feature_sensitivities = top_patient_feature_sensitivities(
            input_df
        )

        result = "CKD" if prediction == 1 else "Not CKD"

        return jsonify({
            "prediction": result,
            "ckd_probability": round(ckd_probability * 100, 2),
            "top_feature_sensitivities": feature_sensitivities,
            "explanation_note": (
                "One-feature-at-a-time sensitivity against training medians "
                "and modes; not a causal explanation."
            ),
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/batch-predict", methods=["POST"])
def batch_predict():

    try:
        if "file" not in request.files:
            return jsonify({"error": "Please upload a CSV file"}), 400

        file = request.files["file"]


        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        # Read uploaded CSV
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "CSV file is empty"}), 400

        df.columns = (
            df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        )
        if df.columns.duplicated().any():
            return jsonify({"error": "CSV contains duplicate column names"}), 400
        df = df.drop(columns=["id", "classification"], errors="ignore")

        # Validate required features
        missing = [
            feature for feature in feature_names
            if feature not in df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        # Select features in the exact training order
        input_df = df[feature_names].copy()

        # Validate numerical features
        for feature in numeric_features:
            input_df[feature] = input_df[feature].replace("?", pd.NA)
            input_df[feature] = pd.to_numeric(
                input_df[feature], errors="raise"
            )

        # Normalize categorical values
        categorical_features = [
            feature for feature in feature_names
            if feature not in numeric_features
        ]

        for feature in categorical_features:
            input_df[feature] = (
                input_df[feature]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"?": pd.NA, "": pd.NA})
            )

        # Validate categorical values
        allowed_values = {
            "rbc": ["normal", "abnormal"],
            "pc": ["normal", "abnormal"],
            "pcc": ["present", "notpresent"],
            "ba": ["present", "notpresent"],
            "htn": ["yes", "no"],
            "dm": ["yes", "no"],
            "cad": ["yes", "no"],
            "appet": ["good", "poor"],
            "pe": ["yes", "no"],
            "ane": ["yes", "no"]
        }

        for feature, allowed in allowed_values.items():
            invalid = input_df[feature].notna() & ~input_df[feature].isin(allowed)

            if invalid.any():
                return jsonify({
                    "error": f"Invalid values in {feature}",
                    "allowed_values": allowed
                }), 400

        # Batch prediction using existing pipeline
        predictions = model.predict(input_df)

        probabilities = model.predict_proba(input_df)

        classes = list(model.classes_)
        ckd_index = classes.index(1)

        results = []

        for i, prediction in enumerate(predictions):

            ckd_probability = float(probabilities[i][ckd_index])

            results.append({
                "patient_number": i + 1,
                "prediction": "CKD" if prediction == 1 else "Not CKD",
                "ckd_probability": round(
                    ckd_probability * 100, 2
                )
            })

        # Summary
        total = len(results)
        ckd_count = sum(
            item["prediction"] == "CKD" for item in results
        )
        non_ckd_count = total - ckd_count

        return jsonify({
            "total_patients": total,
            "ckd_count": ckd_count,
            "non_ckd_count": non_ckd_count,
            "results": results
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV contains no readable data"}), 400

    except (ValueError, TypeError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
