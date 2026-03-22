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
# Clean dataset
# -----------------------------
df = df[df["session_id"].str.contains("session")]

valid_gestures = ["fist", "four", "index", "open", "small"]
df = df[df["gesture_label"].isin(valid_gestures)]

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = []

for i in range(21):
    feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

# -----------------------------
# Evaluate per session
# -----------------------------
sessions = sorted(df["session_id"].unique())

results = []

print("\n=== Session-wise Accuracy ===")

for session in sessions:

    session_df = df[df["session_id"] == session]

    X = session_df[feature_cols]
    y = session_df["gesture_label"]

    if len(X) == 0:
        continue

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    print(f"{session} → Accuracy: {acc:.4f}")

    results.append((session, acc))

# -----------------------------
# Summary
# -----------------------------
accuracies = [r[1] for r in results]

print("\n=== Summary ===")
print("Min Accuracy:", np.min(accuracies))
print("Max Accuracy:", np.max(accuracies))
print("Average Accuracy:", np.mean(accuracies))