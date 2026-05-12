CREATE DATABASE IF NOT EXISTS heart_health;
USE heart_health;

-- ---------------- USERS TABLE ----------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    age INT NOT NULL,
    gender ENUM('male','female','other') NOT NULL,
    medical_history TEXT
);

-- ---------------- ECG REPORTS TABLE ----------------
CREATE TABLE IF NOT EXISTS ecg_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,

    -- Prediction Info
    pred_class INT,
    confidence FLOAT,
    risk_level VARCHAR(20),
    message TEXT,

    -- File Info
    file_type VARCHAR(20),
    ecg_image VARCHAR(255),

    -- ECG Parameters
    heart_rate FLOAT,
    pr_interval FLOAT,
    qrs_duration FLOAT,
    qt_interval FLOAT,

    -- Timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);