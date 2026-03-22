import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = r"C:\Users\91709\Documents\HandGestureProject\data\processed_landmarks.csv"
MODEL_PATH = r"C:\Users\91709\Documents\HandGestureProject\models\gesture_model.pkl"
CONF_MATRIX_PATH = r"C:\Users\91709\Documents\HandGestureProject\results\confusion_matrix.png"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Total samples:", len(df))
print("Columns:", df.columns)

# -----------------------------
# DATASET CLEANING
# -----------------------------

# Remove invalid sessions (like WIN)
df = df[df["session_id"].str.contains("session")]

# Keep only valid gesture labels
valid_gestures = ["fist", "four", "index", "open", "small"]

df = df[df["gesture_label"].isin(valid_gestures)]

print("Samples after cleaning:", len(df))

# Show class distribution
print("\nGesture distribution:")
print(df["gesture_label"].value_counts())

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = []

for i in range(21):
    feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

# -----------------------------
# Session-based split
# -----------------------------
sessions = sorted(df["session_id"].unique())

print("\nSessions found:", sessions)

train_sessions = sessions[:-2]
test_sessions = sessions[-2:]

print("Train sessions:", train_sessions)
print("Test sessions:", test_sessions)

train_df = df[df["session_id"].isin(train_sessions)]
test_df = df[df["session_id"].isin(test_sessions)]

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

# -----------------------------
# Train/Test sets
# -----------------------------
X_train = train_df[feature_cols]
y_train = train_df["gesture_label"]

X_test = test_df[feature_cols]
y_test = test_df["gesture_label"]

# -----------------------------
# Train RandomForest
# -----------------------------
print("\nTraining RandomForest...")

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=25,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Save confusion matrix
# -----------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gesture Confusion Matrix")

plt.savefig(CONF_MATRIX_PATH)

print("\nConfusion matrix saved:", CONF_MATRIX_PATH)

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)