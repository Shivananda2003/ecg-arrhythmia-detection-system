import numpy as np
import arrhythmia

# Load dataset
X = np.load("X_test.npy")
y = np.load("y_test.npy")

correct = 0
total = len(X)

all_preds = []

for i in range(total):
    sample = X[i].flatten()
    text = ",".join(map(str, sample))

    result = arrhythmia.predict_single_beat(text)

    pred = result["pred_class"]
    actual = int(y[i])

    all_preds.append(pred)

    if pred == actual:
        correct += 1

    # print few samples
    if i < 5:
        print(f"\nSample {i}")
        print("Predicted:", pred)
        print("Actual:", actual)

accuracy = correct / total

print("\n====================")
print("Total Samples:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)