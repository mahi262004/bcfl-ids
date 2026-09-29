"""
Centralized IDS baseline -- the PERT-chart stage that comes BEFORE
Federated IDS. This trains one model on all 9 devices' data pooled
together, giving a reference accuracy to compare the federated
version against later (federated models are expected to trade a
little accuracy for privacy -- you need this number to show that
trade-off honestly in your results section).

Usage:
    python fl/centralized_baseline.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fl.data_loader import NON_FEATURE_COLUMNS, build_centralized_dataset

MODEL_OUT_PATH = "data/processed/centralized_baseline_model.joblib"
SCALER_OUT_PATH = "data/processed/centralized_baseline_scaler.joblib"

# Keep this small for your first real run -- N-BaIoT has 7M+ rows,
# far more than needed to prove the pipeline works. Increase once
# you've confirmed everything runs end-to-end.
SAMPLE_FRAC = 0.05


def prepare_features(df: pd.DataFrame, label_col: str = "binary_label"):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values
    return X, y, feature_cols


def train_and_evaluate(label_col: str = "binary_label"):
    print(f"Loading N-BaIoT (sample_frac={SAMPLE_FRAC})...")
    df = build_centralized_dataset(sample_frac=SAMPLE_FRAC)
    print(f"Loaded {len(df)} rows across {df['device_id'].nunique()} devices")
    print(f"Label distribution:\n{df[label_col].value_counts()}\n")

    X, y, feature_cols = prepare_features(df, label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Centralized IDS Baseline Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_OUT_PATH)
    joblib.dump(scaler, SCALER_OUT_PATH)
    print(f"\nSaved model to {MODEL_OUT_PATH}")
    print(f"Saved scaler to {SCALER_OUT_PATH}")
    print("(the XAI stage will load these to explain individual predictions)")

    return model, scaler, feature_cols


if __name__ == "__main__":
    train_and_evaluate(label_col="binary_label")
