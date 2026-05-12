import numpy as np
import arrhythmia

# Load test data
X = np.load(r"D:\Workshop\Minor Project(updated2)\X_test.npy")
y = np.load(r"D:\Workshop\Minor Project(updated2)\y_test.npy")

# Take one sample
sample = X[0]
label = y[0]

# Fix shape
sample = sample.flatten()

# Convert to text (same as your UI input)
text = ",".join(map(str, sample))

# Predict
result = arrhythmia.predict_single_beat(text)

print("Expected:", label)
print("Predicted:", result["label"])
print("Full Output:", result)