"""Konfigurasi aplikasi."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = "ganti-dengan-string-acak-yang-panjang-sebelum-deploy"
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB untuk upload CSV

    # Path
    DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "knn_model.pkl"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"
    METRICS_PATH = MODEL_DIR / "metrics.json"

    # Parameter KNN
    DEFAULT_K = 5
    FEATURES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    # Kolom yang tidak boleh bernilai 0 secara medis
    ZERO_AS_MISSING = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    ]