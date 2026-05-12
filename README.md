# ECG Arrhythmia Detection System using CNN

A collaborative healthcare-focused web application developed for ECG arrhythmia analysis and prediction using deep learning techniques. The system provides ECG waveform visualization, prediction workflows, and interactive analysis features through a Flask-based web interface.

---

## Features

* ECG waveform visualization
* CNN-based arrhythmia prediction workflow
* File upload support for ECG data/images
* Dynamic graph rendering for ECG signals
* Interactive frontend interface
* Flask backend integration
* Healthcare-oriented monitoring workflow

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask
* Python

### Libraries & Tools

* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* WFDB
* NeuroKit2

---

## My Contribution

* Worked on frontend development and UI integration
* Contributed to Flask backend integration and workflow handling
* Implemented ECG waveform visualization and graph rendering features
* Assisted in file upload workflows and prediction result handling

---

## Project Structure

```bash
static/                 # CSS, images, videos
templates/              # HTML templates
app.py                  # Main Flask application
arrhythmia.py           # Prediction-related logic
ecg_image_processor.py  # ECG image processing
evaluate_model.py       # Model evaluation
test_model.py           # Testing scripts
schema.sql              # Database schema
requirements.txt        # Project dependencies
```

---

## Installation

```bash
git clone https://github.com/Shivananda2003/ecg-arrhythmia-detection-system.git

cd ecg-arrhythmia-detection-system

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

---

## Note

Large dataset files, trained model weights, and generated test files were excluded from this repository due to storage limitations.
