import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import accuracy_score

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = r"C:\Users\91709\Documents\HandGestureProject\data\processed_landmarks.csv"
MODEL_PATH = r"C:\Users\91709\Documents\HandGestureProject\models\gesture_model.pkl"

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("Total samples:", len(df))

# -----------------------------
# Clean dataset (IMPORTANT)
# -----------------------------
df = df[df["session_id"].str.contains("session")]

valid_gestures = ["fist", "four", "index", "open", "small"]
df = df[df["gesture_label"].isin(valid_gestures)]

print("Samples after cleaning:", len(df))

# -----------------------------
# Load trained model
# -----------------------------
print("\nLoading model...")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = []

for i in range(21):
    feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

# -----------------------------
# Define conditions (EDIT IF NEEDED)
# -----------------------------
scenarios = {
    "Controlled Environment": [
        "session01", "session02", "session03", "session04", "session05"
    ],
    "Moderate Variation": [
        "session06", "session07", "session08", "session09", "session10", "session11"
    ],
    "Challenging Condition": [
        "session12"
    ]
}

# -----------------------------
# Evaluate each condition
# -----------------------------
results = []

print("\n=== Condition-wise Evaluation ===")

for scenario, sessions in scenarios.items():

    test_df = df[df["session_id"].isin(sessions)]

    X_test = test_df[feature_cols]
    y_test = test_df["gesture_label"]

    if len(X_test) == 0:
        print(f"⚠ No data for {scenario}")
        continue

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{scenario} → Accuracy: {acc:.4f}")

    results.append((scenario, acc))

# -----------------------------
# Summary
# -----------------------------
print("\n=== FINAL SUMMARY ===")

accuracies = [r[1] for r in results]

print("Min Accuracy:", np.min(accuracies))
print("Max Accuracy:", np.max(accuracies))
print("Average Accuracy:", np.mean(accuracies))