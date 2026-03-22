import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# =============================
# Paths
# =============================

DATASET_PATH = r"C:\Users\91709\Documents\HandGestureProject\dataset\raw_videos"
OUTPUT_CSV = r"C:\Users\91709\Documents\HandGestureProject\data\processed_landmarks.csv"

print("Scanning dataset...")

# =============================
# Find all videos
# =============================

video_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".mp4"):
            video_files.append(os.path.join(root, file))

print("Videos found:", len(video_files))

if len(video_files) == 0:
    print("❌ No videos found")
    exit()

# =============================
# Mediapipe Hands
# =============================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =============================
# CSV Columns
# =============================

columns = ["session_id", "gesture_label"]

for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]

rows = []

# =============================
# Process videos
# =============================

for video_path in video_files:

    video_name = os.path.basename(video_path)

    # Expected name format: session01_fist_1.mp4
    parts = video_name.replace(".mp4","").split("_")

    if len(parts) < 2:
        print("Skipping invalid name:", video_name)
        continue

    session_id = parts[0]
    gesture_label = parts[1]

    print("\nProcessing:", video_name)

    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    detections = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame,(640,480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            detections += 1

            hand = results.multi_hand_landmarks[0]

            # =============================
            # Extract landmarks
            # =============================

            landmarks = []

            for lm in hand.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # =============================
            # NORMALIZATION STEP
            # =============================

            # Wrist normalization
            landmarks = landmarks - landmarks[0]

            # Scale normalization
            max_value = np.max(np.abs(landmarks))

            if max_value != 0:
                landmarks = landmarks / max_value

            # Flatten landmarks
            flattened = landmarks.flatten()

            row = [session_id, gesture_label] + flattened.tolist()

            rows.append(row)

        frame_index += 1

    cap.release()

    print("Frames:", frame_index)
    print("Detections:", detections)

print("\nTotal rows collected:", len(rows))

# =============================
# Save CSV
# =============================

df = pd.DataFrame(rows, columns=columns)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)

print("\nCSV saved:", OUTPUT_CSV)