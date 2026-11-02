"""
Script training model KNN dengan StandardScaler.
Jalankan dari root projek: python -m ml.train
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Agar bisa import config saat dijalankan sebagai modul
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config  # noqa: E402


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ganti nilai 0 pada kolom medis dengan median kolom (di luar 0)."""
    df = df.copy()
    for col in Config.ZERO_AS_MISSING:
        if col in df.columns:
            median = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median)
    return df


def main():
    print("Memuat dataset...")
    df = pd.read_csv(Config.DATA_PATH)
    print(f"  Total baris: {len(df)}, kolom: {list(df.columns)}")

    df = clean_missing_values(df)
    print("  Nilai 0 pada kolom medis sudah diganti median.")

    X = df[Config.FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} baris, Test: {len(X_test)} baris")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  StandardScaler sudah di-fit pada data train.")

    print(f"Melatih KNN dengan K={Config.DEFAULT_K}...")
    model = KNeighborsClassifier(n_neighbors=Config.DEFAULT_K)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

    print()
    print("=" * 50)
    print(f"Akurasi : {acc * 100:.2f}%")
    print(f"Presisi : {prec * 100:.2f}%")
    print(f"Recall  : {rec * 100:.2f}%")
    print("=" * 50)
    print()
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["Tidak Diabetes", "Diabetes"]
    ))

    Config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)

    metrics = {
        "akurasi": round(float(acc) * 100, 2),
        "presisi": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(df)),
        "k": Config.DEFAULT_K,
        "cm": {
            "TP": int(cm[0][0]),
            "FN": int(cm[0][1]),
            "FP": int(cm[1][0]),
            "TN": int(cm[1][1]),
        },
        "features": Config.FEATURES,
    }
    Config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model, scaler, dan metrik tersimpan di {Config.MODEL_DIR}")


if __name__ == "__main__":
    main()