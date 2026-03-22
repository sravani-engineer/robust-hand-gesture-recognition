import pandas as pd
import os

INPUT_CSV = r"C:\Users\91709\Documents\HandGestureProject\data\hand_landmarks.csv"

df = pd.read_csv(INPUT_CSV)

print("Original rows:", len(df))
print(df.head())


# STEP 2 — Extract session id
df["session_id"] = df["video_name"].apply(lambda x: x.split("_")[0])


# STEP 3 — Extract gesture label
df["gesture_label"] = df["video_name"].apply(lambda x: x.split("_")[1])


# STEP 4 — Normalize landmarks
for i in range(21):

    df[f"x{i}"] = df[f"x{i}"] - df["x0"]
    df[f"y{i}"] = df[f"y{i}"] - df["y0"]
    df[f"z{i}"] = df[f"z{i}"] - df["z0"]


print("Landmarks normalized")


# STEP 5 — Drop unused columns
df = df.drop(columns=["video_name", "frame_index"])


# STEP 6 — Save processed dataset
OUTPUT_CSV = r"C:\Users\91709\Documents\HandGestureProject\data\processed_landmarks.csv"

df.to_csv(OUTPUT_CSV, index=False)

print("Processed dataset saved at:", OUTPUT_CSV)