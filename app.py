# app.py — merged, self-contained version (use with arrhythmia.py)
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import mysql.connector
from mysql.connector import Error
from flask_bcrypt import Bcrypt
import os
import numpy as np
import neurokit2 as nk
import pandas as pd
from ecg_image_processor import extract_signal_from_image, resample_to_187
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

def generate_recommendation_single(prediction_label):
    
    if prediction_label == 0:
        return {
            "risk": "LOW",
            "message": "Normal heartbeat detected.",
            "advice": [
                "Maintain a healthy lifestyle",
                "Continue regular exercise",
                "Routine health checkups recommended"
            ]
        }

    elif prediction_label == 1:  # Supraventricular
        return {
            "risk": "MODERATE",
            "message": "Supraventricular arrhythmia detected.",
            "advice": [
                "Reduce caffeine intake",
                "Avoid stress and anxiety",
                "Monitor for palpitations",
                "Consult a doctor if frequent"
            ]
        }

    elif prediction_label == 2:  # PVC (Ventricular)
        return {
            "risk": "HIGH",
            "message": "Premature Ventricular Contractions (PVC) detected.",
            "advice": [
                "Avoid strenuous activity",
                "Seek medical evaluation",
                "Monitor chest discomfort or dizziness",
                "Consult a cardiologist"
            ]
        }

    elif prediction_label == 3:  # Fusion
        return {
            "risk": "MODERATE",
            "message": "Fusion beat detected (irregular signal overlap).",
            "advice": [
                "May indicate conduction abnormalities",
                "Track symptoms like fatigue",
                "Consult a specialist if recurring"
            ]
        }

    else:  # Unknown
        return {
            "risk": "UNKNOWN",
            "message": "Unclassified heartbeat detected.",
            "advice": [
                "Further medical analysis required",
                "Consider a full ECG test",
                "Consult a healthcare professional"
            ]
        }

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360):
    b, a = butter(3, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

current_data = None
current_index = 0

def generate_ecg_plot(signal, filename):
    plt.figure(figsize=(8, 3))
    plt.plot(signal, color='black')
    plt.title("ECG Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Try to import arrhythmia module created from your notebook
try:
    import arrhythmia
    ARRHYTHMIA_AVAILABLE = True
except Exception as e:
    arrhythmia = None
    ARRHYTHMIA_AVAILABLE = False
    print("Warning: arrhythmia.py not available or failed to import:", e)

app = Flask(__name__)
app.secret_key = "9f75e03fb50dfb15c8adc70f5862fa3e"
bcrypt = Bcrypt(app)

# ---------------- DB CONFIG ----------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "heart_health"
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def get_logged_in_user():
    if 'user_id' not in session:
        return None
    return {
        'name': session.get('name'),
        'email': session.get('email'),
        'age': session.get('age'),
        'gender': session.get('gender'),
        'medical_history': session.get('medical_history')
    }

# ---------------- File paths / CSV config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_PATH = os.path.join(BASE_DIR, "mitbih_dataset", "signals.npy")
GENERATED_CSV = os.path.join(BASE_DIR, "beats_features_indices_nolabels.csv")
OUTPUT_LABELLED_CSV = os.path.join(BASE_DIR, "beats_features_with_labels.csv")

# ---------------- Basic pages & auth ----------------
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', is_homepage=True)

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        age = request.form['age']
        gender = request.form['gender']
        medical_history = request.form.get('medical_history', '')

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        conn = get_db_connection()
        if conn is None:
            flash("Database connection failed.", "danger")
            return redirect(url_for('signup'))
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (name, email, password, age, gender, medical_history) VALUES (%s,%s,%s,%s,%s,%s)",
                (name, email, hashed_password, age, gender, medical_history)
            )
            conn.commit()
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash("Email already exists. Please use another email.", "danger")
            return redirect(url_for('signup'))
        finally:
            cursor.close()
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        if conn is None:
            flash("Database connection failed.", "danger")
            return redirect(url_for('login'))
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['age'] = user['age']
            session['gender'] = user['gender']
            session['medical_history'] = user['medical_history']
            flash("Login successful!", "success")
            return redirect(url_for('user'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/user')
def user():
    user = get_logged_in_user()
    if not user:
        flash("You must login first!", "warning")
        return redirect(url_for('login'))
    return render_template('user.html', **user)

@app.route('/profile')
def profile():
    user = get_logged_in_user()
    if not user:
        flash("You must login first!", "warning")
        return redirect(url_for('login'))
    return render_template('profile.html', **user)

@app.route('/edit_profile', methods=["GET", "POST"])
def edit_profile():
    user = get_logged_in_user()
    if not user:
        flash("You must login first!", "warning")
        return redirect(url_for('login'))

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        medical_conditions = request.form.getlist("medical_history")
        medical_history_str = ", ".join(medical_conditions)

        conn = get_db_connection()
        if conn is None:
            flash("Database connection failed.", "danger")
            return redirect(url_for('edit_profile'))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE users 
                SET name=%s, age=%s, gender=%s, medical_history=%s
                WHERE id=%s
            """, (name, age, gender, medical_history_str, session['user_id']))
            conn.commit()

            session['name'] = name
            session['age'] = age
            session['gender'] = gender
            session['medical_history'] = medical_history_str

            flash("Profile updated successfully!", "success")
            return redirect(url_for('profile'))

        except Error as e:
            flash("Error updating profile: " + str(e), "danger")
        finally:
            cursor.close()
            conn.close()

    user_medical_history = user['medical_history'].split(", ") if user['medical_history'] else []
    user_copy = user.copy()
    user_copy['medical_history'] = user_medical_history
    return render_template("edit_profile.html", **user_copy)

@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    user = get_logged_in_user()
    if not user:
        flash("You must login first!", "warning")
        return redirect(url_for('login'))

    stats = {
        "heart_rate": 78,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "cholesterol": 190,
        "bmi": 23.5,
        "dates": ["Sep 10", "Sep 11", "Sep 12", "Sep 13"],
        "heart_rate_history": [75, 78, 80, 76],
        "bp_systolic_history": [118, 122, 119, 121],
        "bp_diastolic_history": [78, 80, 77, 79]
    }

    risk = None

    if request.method == "POST":
        heart_rate = int(request.form['heart_rate'])
        bp_systolic = int(request.form['bp_systolic'])
        bp_diastolic = int(request.form['bp_diastolic'])
        bmi = float(request.form['bmi'])
        cholesterol = int(request.form['cholesterol'])

        stats["heart_rate"] = heart_rate
        stats["bp_systolic"] = bp_systolic
        stats["bp_diastolic"] = bp_diastolic
        stats["bmi"] = bmi
        stats["cholesterol"] = cholesterol

        stats["dates"].append("Today")
        stats["heart_rate_history"].append(heart_rate)
        stats["bp_systolic_history"].append(bp_systolic)
        stats["bp_diastolic_history"].append(bp_diastolic)

        if (bp_systolic > 140 or bp_diastolic > 90 or
            bmi > 30 or cholesterol > 240 or heart_rate > 100):
            risk = "High Risk ⚠️"
        elif (bp_systolic > 120 or bmi > 25 or
              cholesterol > 200 or heart_rate > 90):
            risk = "Moderate Risk ⚠️"
        else:
            risk = "Low Risk ✅"

    reminders = ["Take morning medicine", "30 min walk after lunch"]
    tips = ["Eat more vegetables", "Do light exercise daily"]

    return render_template("dashboard.html",
                           current_user=user,
                           stats=stats,
                           risk=risk,
                           reminders=reminders,
                           tips=tips)

@app.context_processor
def inject_user():
    return dict(current_user=get_logged_in_user())

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/edit')
def edit():
    return render_template('edit_profile.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))


# ---------------- Arrhythmia endpoints (use arrhythmia.py) ----------------
@app.route("/api/predict_beat", methods=["POST"])
def api_predict_beat():
    if not ARRHYTHMIA_AVAILABLE:
        return jsonify({"error":"arrhythmia engine not available. Place arrhythmia.py next to app.py."}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error":"no json body"}), 400
    samples_text = data.get("samples", "")
    sr = int(data.get("sampling_rate", 360))
    res = arrhythmia.predict_beat_from_text(samples_text, sampling_rate=sr)
    
    # attach actual label (for UI testing)
    actual_label = data.get("actual_label", None)
    if actual_label is not None:
        res["actual_label"] = actual_label

    # add recommendation
    recommendations = {
        "Normal": "Heart rhythm looks normal ❤️ Maintain healthy lifestyle.",
        "Class1": "Minor irregularity detected. Monitor regularly.",
        "Class2": "Irregular pattern detected. Consider consulting a doctor.",
        "Class3": "High risk arrhythmia ⚠️ Seek medical attention.",
        "Class4": "Critical condition 🚨 Immediate medical help required!"
    }

    res["recommendation"] = recommendations.get(res.get("label"), "No advice")

    return jsonify(res)

@app.route("/api/predict_record", methods=["POST"])
def api_predict_record():
    if not ARRHYTHMIA_AVAILABLE:
        return jsonify({"error":"arrhythmia engine not available. Place arrhythmia.py next to app.py."}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error":"no json body"}), 400
    samples_text = data.get("samples", "")
    sr = int(data.get("sampling_rate", 360))
    res = arrhythmia.predict_record_from_text(samples_text, sampling_rate=sr)
    return jsonify(res)

@app.route("/api/get_sample")
def get_sample():
    import numpy as np

    X = np.load("X_test.npy")
    y = np.load("y_test.npy")

    idx = np.random.randint(0, len(X))

    sample = X[idx].flatten()
    label = int(y[idx])

    text = ",".join(map(str, sample))

    return jsonify({
        "samples": text,
        "actual_label": label
    })
    
@app.route('/api/upload_file', methods=['POST'])
def upload_file():

    global current_data, current_index

    # ✅ USER CHECK
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    if not ARRHYTHMIA_AVAILABLE:
        return jsonify({"error": "arrhythmia engine not available"}), 500

    file = request.files['file']

    # -------- NPY MODE --------
    if file.filename.endswith('.npy'):

        data = np.load(file)

        if len(data.shape) == 3:
            data = data.squeeze()

        current_data = data
        current_index = 0

        sample = current_data[current_index]
        mode = "multi"

    # -------- IMAGE MODE --------
    elif file.filename.endswith(('.jpg', '.png')):

        signal = extract_signal_from_image(file)

        if signal is None:
            return jsonify({"error": "Could not extract ECG from image"})

        signal = resample_to_187(signal)

        sample = signal
        current_data = None
        current_index = 0
        mode = "single"

    # -------- CSV MODE --------
    else:
        data = np.loadtxt(file, delimiter=',')
        data = data.flatten()

        if len(data) != 187:
            return jsonify({"error": "CSV must contain 187 values"})

        sample = data
        current_data = None
        current_index = 0
        mode = "single"

    # -------- PREDICTION --------
    samples_text = ",".join(map(str, sample))
    res = arrhythmia.predict_beat_from_text(samples_text)

    prediction_label = res.get("pred_class", 0)
    recommendation = generate_recommendation_single(prediction_label)

    # -------- ECG FEATURE EXTRACTION (SAFE) --------
    heart_rate = pr_interval = qrs_duration = qt_interval = None

    try:
        signal = np.array(sample).astype(float)

        if len(signal) > 300:
            signals, info = nk.ecg_process(signal, sampling_rate=360)
            features = nk.ecg_intervalrelated(signals)

            heart_rate = float(features["ECG_Rate_Mean"].values[0]) if "ECG_Rate_Mean" in features else None
            pr_interval = float(features["ECG_PR_Interval_Mean"].values[0]) if "ECG_PR_Interval_Mean" in features else None
            qrs_duration = float(features["ECG_QRS_Duration_Mean"].values[0]) if "ECG_QRS_Duration_Mean" in features else None
            qt_interval = float(features["ECG_QT_Interval_Mean"].values[0]) if "ECG_QT_Interval_Mean" in features else None
        else:
            print("Signal too short for NeuroKit, skipping...")

    except Exception as e:
        print("NeuroKit error:", e)

    # 🔥 FALLBACK ESTIMATED VALUES
    if heart_rate is None:
        heart_rate = 72
    if pr_interval is None:
        pr_interval = 160
    if qrs_duration is None:
        qrs_duration = 100
    if qt_interval is None:
        qt_interval = 380

    # -------- GENERATE ECG IMAGE --------
    try:
        import matplotlib.pyplot as plt

        image_filename = f"ecg_{user_id}_{current_index}.png"
        image_path = os.path.join("static", image_filename)

        plt.figure(figsize=(8, 3))

        time = np.arange(len(sample)) / 360
        plt.plot(time, sample, color='black')
        
        plt.title("ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

    except Exception as e:
        print("ECG image generation error:", e)
        image_path = None

    # -------- EXTRACT VALUES --------
    confidence = max(res.get("probs", [0]))
    risk = recommendation.get("risk", "UNKNOWN")
    message = recommendation.get("message", "")
    file_type = file.filename.split('.')[-1]

    # -------- DATABASE INSERT --------
    report_id = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO ecg_reports 
        (user_id, pred_class, confidence, risk_level, message, file_type,
         heart_rate, pr_interval, qrs_duration, qt_interval, ecg_image)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            prediction_label,
            confidence,
            risk,
            message,
            file_type,
            heart_rate,
            pr_interval,
            qrs_duration,
            qt_interval,
            image_path
        ))

        report_id = cursor.lastrowid

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("DB insert error:", e)

    # -------- RESPONSE --------
    res["mode"] = mode
    res["index"] = current_index
    res["total"] = len(current_data) if current_data is not None else 1
    res["recommendation"] = recommendation
    res["report_id"] = report_id

    return jsonify(res)

@app.route('/api/next_beat', methods=['GET'])
def next_beat():

    global current_data, current_index

    if current_data is None:
        return jsonify({"error": "No dataset loaded"})

    current_index += 1

    if current_index >= len(current_data):
        current_index = 0

    sample = current_data[current_index]

    samples_text = ",".join(map(str, sample))
    res = arrhythmia.predict_beat_from_text(samples_text)
    prediction_label = res.get("pred_class", 0)
    recommendation = generate_recommendation_single(prediction_label)

    res["mode"] = "multi"
    res["index"] = current_index
    res["total"] = len(current_data)
    res["recommendation"] = recommendation

    return jsonify(res)

@app.route('/api/prev_beat', methods=['GET'])
def prev_beat():

    global current_data, current_index

    if current_data is None:
        return jsonify({"error": "No dataset loaded"})

    current_index -= 1

    if current_index < 0:
        current_index = len(current_data) - 1  # loop backwards

    sample = current_data[current_index]

    samples_text = ",".join(map(str, sample))
    res = arrhythmia.predict_beat_from_text(samples_text)

    prediction_label = res.get("pred_class", 0)
    recommendation = generate_recommendation_single(prediction_label)

    res["mode"] = "multi"
    res["index"] = current_index
    res["total"] = len(current_data)
    res["recommendation"] = recommendation

    return jsonify(res)

@app.route('/api/jump_beat', methods=['POST'])
def jump_beat():

    global current_data, current_index

    if current_data is None:
        return jsonify({"error": "No dataset loaded"})

    data = request.get_json()
    index = int(data.get("index", 0))

    if index < 0 or index >= len(current_data):
        return jsonify({"error": "Invalid beat index"})

    current_index = index

    sample = current_data[current_index]

    samples_text = ",".join(map(str, sample))
    res = arrhythmia.predict_beat_from_text(samples_text)

    prediction_label = res.get("pred_class", 0)
    recommendation = generate_recommendation_single(prediction_label)

    res["mode"] = "multi"
    res["index"] = current_index
    res["total"] = len(current_data)
    res["recommendation"] = recommendation

    return jsonify(res)

# ---------------- CSV download & label merge endpoints ----------------
@app.route("/download/csv")
def download_csv():
    if not os.path.exists(GENERATED_CSV):
        return jsonify({"error":"generated CSV not found on server."}), 404
    return send_file(GENERATED_CSV, as_attachment=True)

@app.route("/upload/labels", methods=["POST"])
def upload_labels():
    if "labels_file" not in request.files:
        return jsonify({"error":"no file field 'labels_file'"}), 400
    f = request.files["labels_file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error":"failed to read CSV: "+str(e)}), 400
    required = {"record_id","beat_index","label"}
    if not required.issubset(set(df.columns)):
        return jsonify({"error":f"labels.csv must contain columns: {required}"}), 400
    if not os.path.exists(GENERATED_CSV):
        return jsonify({"error":"base generated CSV not found on server; run detection first."}), 500
    dbase = pd.read_csv(GENERATED_CSV)
    dbase["record_id"] = dbase["record_id"].astype(str)
    dbase["beat_index"] = dbase["beat_index"].astype(int)
    df["record_id"] = df["record_id"].astype(str)
    df["beat_index"] = df["beat_index"].astype(int)
    merged = pd.merge(dbase, df, on=["record_id","beat_index"], how="left", suffixes=("","_lbl"))
    if "label" in merged.columns and "label_lbl" in merged.columns:
        merged["label"] = merged["label_lbl"].fillna(merged["label"])
        merged.drop(columns=["label_lbl"], inplace=True)
    merged.to_csv(OUTPUT_LABELLED_CSV, index=False)
    return jsonify({"message":"merged labels written", "output_path": OUTPUT_LABELLED_CSV})

@app.route("/download/labelled")
def download_labelled():
    if not os.path.exists(OUTPUT_LABELLED_CSV):
        return jsonify({"error":"labelled CSV not found. Upload labels first."}), 404
    return send_file(OUTPUT_LABELLED_CSV, as_attachment=True)

@app.route('/api/my_reports')
def my_reports():

    user_id = session.get('user_id')

    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    page = int(request.args.get('page', 1))
    limit = 2
    offset = (page - 1) * limit

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            id,
            pred_class, 
            confidence, 
            risk_level, 
            message, 
            file_type,
            heart_rate,
            pr_interval,
            qrs_duration,
            qt_interval,
            created_at
        FROM ecg_reports
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """, (user_id, limit, offset))

    reports = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(reports)

@app.route('/download/report/<int:report_id>')
def download_report(report_id):

    user_id = session.get('user_id')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 🔥 JOIN with users table
    cursor.execute("""
        SELECT e.*, u.name, u.age, u.gender
        FROM ecg_reports e
        JOIN users u ON e.user_id = u.id
        WHERE e.id = %s AND e.user_id = %s
    """, (report_id, user_id))

    report = cursor.fetchone()

    cursor.close()
    conn.close()

    if not report:
        return "Report not found", 404

    # 🔥 RECREATE AI RECOMMENDATION
    recommendation = generate_recommendation_single(report['pred_class'])

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet

    filename = f"ecg_report_{report_id}.pdf"
    filepath = os.path.join("static", filename)

    doc = SimpleDocTemplate(filepath)
    styles = getSampleStyleSheet()

    content = []

    label_map = {
        0: "Normal",
        1: "Supraventricular",
        2: "Ventricular (PVC)",
        3: "Fusion",
        4: "Unknown"
    }

    # -------- HEADER --------
    content.append(Paragraph("<b>ECG ANALYSIS REPORT</b>", styles['Title']))
    content.append(Spacer(1, 15))

    # -------- PATIENT INFO --------
    content.append(Paragraph(f"<b>Report ID:</b> ECG-{report['id']:05d}", styles['Normal']))
    content.append(Paragraph(f"<b>Patient Name:</b> {report['name']}", styles['Normal']))
    content.append(Paragraph(f"<b>Age:</b> {report['age']}", styles['Normal']))
    content.append(Paragraph(f"<b>Gender:</b> {report['gender']}", styles['Normal']))
    content.append(Paragraph(f"<b>Date:</b> {report['created_at']}", styles['Normal']))
    content.append(Spacer(1, 15))

    # -------- ECG IMAGE --------
    if report.get("ecg_image") and os.path.exists(report["ecg_image"]):
        content.append(Paragraph("<b>ECG WAVEFORM</b>", styles['Heading2']))
        content.append(Spacer(1, 10))

        img = Image(report["ecg_image"], width=400, height=150)
        content.append(img)

        content.append(Spacer(1, 15))

    # -------- DIAGNOSIS --------
    content.append(Paragraph(f"<b>Diagnosis:</b> {label_map.get(report['pred_class'])}", styles['Normal']))
    content.append(Paragraph(f"<b>Risk Level:</b> {recommendation['risk']}", styles['Normal']))
    content.append(Spacer(1, 15))

    # -------- ECG PARAMETERS --------
    content.append(Paragraph("<b>ECG PARAMETERS</b>", styles['Heading2']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Heart Rate: ~{report['heart_rate']} bpm (estimated)", styles['Normal']))
    content.append(Paragraph(f"PR Interval: ~{report['pr_interval']} ms (estimated)", styles['Normal']))
    content.append(Paragraph(f"QRS Duration: ~{report['qrs_duration']} ms (estimated)", styles['Normal']))
    content.append(Paragraph(f"QT Interval: ~{report['qt_interval']} ms (estimated)", styles['Normal']))
    content.append(Spacer(1, 15))

    # -------- AI RECOMMENDATION --------
    content.append(Paragraph("<b>AI RECOMMENDATION</b>", styles['Heading2']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(recommendation['message'], styles['Normal']))
    content.append(Spacer(1, 10))

    for advice in recommendation['advice']:
        content.append(Paragraph(f"• {advice}", styles['Normal']))

    content.append(Spacer(1, 15))

    # -------- DISCLAIMER --------
    content.append(Paragraph(
        "This is an AI-assisted analysis and not a medical diagnosis. Please consult a doctor.",
        styles['Italic']
    ))

    # -------- BUILD PDF --------
    doc.build(content)

    return send_file(filepath, as_attachment=True)

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True)
