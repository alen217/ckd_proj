# Chronic Kidney Disease Prediction

## Train and evaluate

From the project directory, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python preprocessing.py
```

Training validates the 400-row, 24-feature dataset, performs a stratified
80/20 split, runs five-fold ROC-AUC evaluation, and writes the selected model
to `kidney_model.joblib`. Evaluation metrics and plots are saved under
`reports/`.

The trained pipelines fit imputation, scaling, and SMOTE within training folds.
Requested accuracy, ROC-AUC, and recall thresholds are reported from measured
results; they are not guaranteed. Feature importance and one-patient
sensitivity summaries are model-specific and are not causal explanations.

## Run the web app

After training, run:

```powershell
python app.py
```

The dashboard supports single-patient predictions and batch CSV screening.
Batch files require the 24 clinical feature columns; an `id` or
`classification` column is ignored. Missing clinical values are imputed by
the trained preprocessing pipeline. Batch results can be downloaded as HTML
or text from the browser.

`python inspectmodel.py` checks the dataset and saved pipeline structure.
`python test_saved_model.py` runs a saved-model prediction smoke test.

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease).

This application is a research/screening aid, not a medical diagnosis or
substitute for a qualified clinician. External validation is required before
any clinical use.
