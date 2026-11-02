"""
Modul prediksi — satu-satunya jalur untuk prediksi & evaluasi.
Dipakai oleh routes manual maupun CSV.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from config import Config


class DiabetesPredictor:
    def __init__(self):
        if not Config.MODEL_PATH.exists() or not Config.SCALER_PATH.exists():
            raise FileNotFoundError(
                "Model belum dilatih. Jalankan: python -m ml.train"
            )
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)

    def get_metrics(self) -> dict:
        if Config.METRICS_PATH.exists():
            return json.loads(Config.METRICS_PATH.read_text(encoding="utf-8"))
        return {}

    def predict_single(self, input_dict: dict):
        """
        Prediksi satu pasien.
        Return: (prediksi:int, label:str, tabel_jarak:list)
        """
        values = [float(input_dict[f]) for f in Config.FEATURES]
        X_raw = np.array(values).reshape(1, -1)
        X_scaled = self.scaler.transform(X_raw)

        pred = int(self.model.predict(X_scaled)[0])
        distances, indices = self.model.kneighbors(X_scaled)

        train_labels = self.model._y
        tabel = []
        for rank, idx in enumerate(indices[0]):
            tabel.append({
                "urutan": rank + 1,
                "jarak": round(float(distances[0][rank]), 6),
                "status": "Diabetes" if train_labels[idx] == 1 else "Tidak Diabetes",
                "k": "Ya" if rank < self.model.n_neighbors else "Tidak",
            })

        label = (
            "Pasien berpotensi mengalami Diabetes."
            if pred == 1
            else "Pasien tidak berpotensi mengalami Diabetes."
        )
        return pred, label, tabel

    def evaluate_csv(self, df_uji: pd.DataFrame) -> dict:
        if "Outcome" not in df_uji.columns:
            raise ValueError("Dataset uji harus memiliki kolom 'Outcome'.")
        missing = [f for f in Config.FEATURES if f not in df_uji.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan: {', '.join(missing)}")

        X = df_uji[Config.FEATURES]
        y = df_uji["Outcome"]
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        cm = confusion_matrix(y, y_pred, labels=[1, 0])
        return {
            "akurasi": round(float(accuracy_score(y, y_pred)) * 100, 2),
            "presisi": round(float(precision_score(y, y_pred, zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)) * 100, 2),
            "jumlah_benar": int((y_pred == y).sum()),
            "jumlah_salah": int((y_pred != y).sum()),
            "n_test": int(len(X)),
            "cm_dict": {
                "TP": int(cm[0][0]),
                "FN": int(cm[0][1]),
                "FP": int(cm[1][0]),
                "TN": int(cm[1][1]),
            },
        }