
import pandas as pd
import joblib

# Import the custom preprocessing class so joblib
# can locate it while loading the saved pipeline.
from ckd_preprocessing import CKDPreprocessor


# Load saved model
artifact = joblib.load("kidney_model.joblib")

model = artifact["model"]
feature_names = artifact["feature_names"]

print("Model loaded successfully!")
print("Number of input features:", len(feature_names))
print("Classifier:", model.named_steps["classifier"])


# Test prediction using one actual dataset record
df = pd.read_csv("kidney_disease.csv")

df.columns = df.columns.str.strip().str.lower()

if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop(columns=["classification"])

# Ensure the input columns match the training order
X = X[feature_names]

sample = X.iloc[[0]]

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\nTest prediction completed!")
print("Predicted class:", prediction)
print("CKD probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Prediction: CKD")
else:
    print("Prediction: Not CKD")
