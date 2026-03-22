import cv2
import mediapipe as mp
import numpy as np
import joblib

# =============================
# Model path
# =============================

MODEL_PATH = r"C:\Users\91709\Documents\HandGestureProject\models\gesture_model.pkl"

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")

# =============================
# Mediapipe Setup
# =============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =============================
# Webcam
# =============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("Webcam started. Press Q to quit.")

# =============================
# Main Loop
# =============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = "No hand"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # =============================
        # Extract landmarks
        # =============================

        landmarks = []

        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # =============================
        # SAME NORMALIZATION AS TRAINING
        # =============================

        landmarks = landmarks - landmarks[0]

        max_value = np.max(np.abs(landmarks))

        if max_value != 0:
            landmarks = landmarks / max_value

        features = landmarks.flatten().reshape(1, -1)

        # =============================
        # Prediction
        # =============================

        prediction = model.predict(features)[0]

        gesture = prediction

    # =============================
    # Display Gesture
    # =============================

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# Cleanup
# =============================

cap.release()
cv2.destroyAllWindows()