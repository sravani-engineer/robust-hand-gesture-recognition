# Robust Hand Gesture Recognition under Real-World Conditions (MediaPipe + ML)

## 📌 Overview
A robust hand gesture recognition system designed to evaluate real-world performance under domain shift conditions such as lighting, background, and distance variation.

Unlike typical gesture projects, this system focuses on generalization and reliability rather than just high accuracy.

💡 This project demonstrates not just model building, but real-world evaluation and failure analysis — key aspects of production-ready ML systems.

---

## 🎯 Problem Statement
Most gesture recognition systems work well in controlled environments but fail under:
- Different lighting conditions
- Background variations
- Distance changes

This project evaluates how well a gesture recognition model performs under such domain shifts.

---

## 🚀 Solution
- Extracted 21 hand landmarks using MediaPipe
- Applied normalization for scale invariance
- Trained a Random Forest classifier
- Performed session-based split to simulate real-world deployment
- Evaluated robustness across multiple conditions

---

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

---

## 📊 Results

### Overall Performance
- Accuracy: **~93–94%**

### Robustness Evaluation

| Condition                | Accuracy |
|-------------------------|----------|
| Controlled Environment  | 1.00     |
| Moderate Variation      | 0.99     |
| Challenging Conditions  | 0.88     |

📌 **Observation:**  
The model achieves near-perfect accuracy in controlled environments but shows a ~12% drop under challenging real-world conditions, highlighting the importance of robustness evaluation.

---

## ⚠️ Failure Analysis

- Confusion observed between **open** and **four** gestures due to similar finger extension patterns  
- Performance drops in challenging conditions (~88%) due to:
  - Reduced landmark stability in low lighting  
  - Lower resolution impact at far distances  
- Model relies heavily on relative finger positions, making it sensitive to partial occlusions  

---

## 🧪 Evaluation Strategy

- Session-based split used instead of random split to avoid data leakage  
- Each session represents different real-world conditions  
- Condition-wise evaluation performed to measure robustness under domain shift  
- Metrics used: Accuracy, Confusion Matrix  

---

## 🧠 Key Learnings
- Model generalization is more important than raw accuracy  
- Landmark normalization significantly improves performance  
- Distance and background affect prediction stability  
- Real-world ML systems must be evaluated beyond standard train-test splits  

---

## 📂 Project Structure
robust-hand-gesture-recognition/
│
├── src/
│ ├── extract_landmarks.py
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate.py
│ ├── evaluate_conditions.py
│ ├── realtime_inference.py
│ ├── test_hand_detection.py
│
├── results/
│ └── confusion_matrix.png
│
├── requirements.txt
├── README.md
├── .gitignore


---

## 💡 What Makes This Project Stand Out

- Uses **session-based splitting** instead of random splitting (prevents data leakage)  
- Evaluates **robustness under domain shift** (lighting, background, distance)  
- Includes **condition-wise performance analysis**  
- Focuses on **real-world reliability**, not just accuracy  

---

## 🎥 Demo

Real-time gesture recognition using webcam.

- Detects hand using MediaPipe  
- Predicts gesture in real-time  
- Works across multiple backgrounds and lighting conditions  

👉 (Demo video will be added soon)

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sravani-engineer/robust-hand-gesture-recognition.git
cd robust-hand-gesture-recognition

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt
### 4. Run real-time inference
python src/realtime_inference.py

## 🔄 Pipeline
Video Input → Raw gesture recordings
Landmark Extraction → MediaPipe (21 keypoints)
Preprocessing → Normalization of coordinates
Model Training → Random Forest classifier
Evaluation → Session-based + condition-wise testing
Deployment → Real-time webcam inference


