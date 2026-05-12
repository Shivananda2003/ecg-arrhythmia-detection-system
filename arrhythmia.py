import os
import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn as nn   # ✅ FIXED (missing import)

# ---------------- MODEL ----------------
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(32*43, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# ---------------- CONFIG ----------------
INPUT_LENGTH = 184
SAMPLING_RATE = 360

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ USE RELATIVE PATH (IMPORTANT)
MODEL_PATH = "ecg_model.pth"

# ---------------- LOAD MODEL ----------------
_model = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        _model = ECG_CNN()  # ✅ create model first
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # ✅ load weights
        _model.to(device)
        _model.eval()
        print(f"[arrhythmia] Loaded model: {MODEL_PATH}")
    except Exception as e:
        print("[arrhythmia] ERROR loading model:", e)
        _model = None
else:
    print("[arrhythmia] No ML model loaded. Using fallback classifier.")


# ---------------- utility: parse text ----------------
def parse_samples_text(text: str):
    toks = text.replace(",", " ").split()
    nums = []
    for t in toks:
        try:
            nums.append(float(t))
        except:
            pass
    return np.array(nums, dtype=np.float32)


# ---------------- prepare beat ----------------
def prepare_beat(arr, input_length=INPUT_LENGTH):
    if arr is None or arr.size == 0:
        return None
    arr = np.array(arr, dtype=np.float32)
    std = arr.std()
    if std < 1e-8:
        arr = arr - arr.mean()
    else:
        arr = (arr - arr.mean()) / (std + 1e-8)
    if arr.size < input_length:
        pad = np.zeros(input_length - arr.size, dtype=np.float32)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:input_length]
    return arr.astype(np.float32)


# ---------------- prediction ----------------
def predict_with_model(beat_np):
    if _model is not None:
        bt = torch.tensor(beat_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = _model(bt)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()

        pred = int(np.argmax(probs))

        # ✅ FIXED: 5-class labels
        labels = ["Normal", "Class1", "Class2", "Class3", "Class4"]

        return {
            "pred_class": pred,
            "label": labels[pred],
            "probs": probs
        }

    # fallback (only if model fails)
    peak = float(np.max(beat_np))
    trough = float(np.min(beat_np))
    ptp = peak - trough
    rms = float(np.sqrt(np.mean(beat_np ** 2)))
    score = ptp / (rms + 1e-8)
    prob_arr = 1 / (1 + np.exp(-0.8 * (score - 3.2)))
    prob_normal = 1 - prob_arr
    probs = [float(prob_normal), float(prob_arr)]
    pred = int(np.argmax(probs))

    return {"pred_class": pred, "label": "Fallback", "probs": probs}


# ---------------- single beat ----------------
def predict_single_beat(samples_text, sr=SAMPLING_RATE):
    arr = parse_samples_text(samples_text)
    if arr.size == 0:
        return {"error": "Invalid or empty samples"}

    beat = prepare_beat(arr, INPUT_LENGTH)
    if beat is None:
        return {"error": "Beat preprocessing failed"}

    out = predict_with_model(beat)
    out["waveform"] = beat.tolist()
    return out


# ---------------- full record ----------------
def predict_record(samples_text, sr=SAMPLING_RATE):
    sig = parse_samples_text(samples_text)
    if sig.size == 0:
        return {"error": "Invalid or empty ECG signal"}

    peaks = find_peaks(sig, distance=int(0.25 * sr))[0]

    results = []
    for p in peaks:
        win = sig[max(0, p - 180): p + 180]
        win = prepare_beat(win, INPUT_LENGTH)
        pred = predict_with_model(win)
        pred.update({"sample_index": int(p)})
        results.append(pred)

    return {"beats": results}


# ---------------- compatibility wrappers ----------------
def predict_beat_from_text(samples_text, sampling_rate=SAMPLING_RATE):
    return predict_single_beat(samples_text, sr=sampling_rate)

def predict_record_from_text(samples_text, sampling_rate=SAMPLING_RATE):
    return predict_record(samples_text, sr=sampling_rate)