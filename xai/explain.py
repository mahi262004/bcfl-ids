"""
Applies SHAP to the trained Centralized IDS baseline to explain
individual predictions -- e.g. which of the 115 traffic features
drove a specific "attack" classification, instead of a black-box
yes/no.

Run from the project root, after fl/centralized_baseline.py has
produced the saved model + scaler:
    python -m xai.explain
"""

import joblib
import numpy as np
import shap

from fl.data_loader import NON_FEATURE_COLUMNS, build_centralized_dataset

MODEL_PATH = "data/processed/centralized_baseline_model.joblib"
SCALER_PATH = "data/processed/centralized_baseline_scaler.joblib"
NUM_SAMPLES_TO_EXPLAIN = 5


def main():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = build_centralized_dataset(sample_frac=0.02)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = scaler.transform(df[feature_cols].values.astype(np.float32))

    background = X[np.random.choice(len(X), size=min(100, len(X)), replace=False)]
    samples_to_explain = X[:NUM_SAMPLES_TO_EXPLAIN]

    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(samples_to_explain)

    print(f"Computed SHAP values for {NUM_SAMPLES_TO_EXPLAIN} sample predictions.")
    print("\nTop contributing features for sample 0 (by absolute SHAP value):")

    # SHAP's return shape differs by version:
    #   - older versions: a list of arrays, one per class -> shap_values[class][sample]
    #   - newer versions: a single array shaped (samples, features, classes)
    # This handles both without depending on which one is installed.
    ATTACK_CLASS_INDEX = 1  # binary_label: 0 = benign, 1 = attack
    if isinstance(shap_values, list):
        contributions = shap_values[ATTACK_CLASS_INDEX][0]
    elif np.ndim(shap_values) == 3:
        contributions = shap_values[0, :, ATTACK_CLASS_INDEX]
    else:
        contributions = shap_values[0]

    top_idx = np.argsort(np.abs(contributions))[::-1][:5]
    for i in top_idx:
        print(f"  {feature_cols[i]}: {contributions[i]:+.4f}")


if __name__ == "__main__":
    main()
