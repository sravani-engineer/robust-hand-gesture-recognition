import matplotlib.pyplot as plt

# -----------------------------
# Data (based on your results)
# -----------------------------
conditions = [
    "Controlled",
    "Moderate Variation",
    "Real-World Conditions"
]

accuracy = [
    1.0000,
    0.9999,
    0.8842
]

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.plot(conditions, accuracy, marker='o')

# Labels
plt.title("Model Performance Across Conditions")
plt.xlabel("Evaluation Condition")
plt.ylabel("Accuracy")

# Grid for clarity
plt.grid()

# Save graph
plt.savefig("results/accuracy_vs_conditions.png")

# Show (optional)
plt.show()