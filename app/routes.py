"""Semua route aplikasi."""
import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.utils import form_to_features, validate_form_input
from ml.predictor import DiabetesPredictor

bp = Blueprint("main", __name__)


def _get_predictor():
    try:
        return DiabetesPredictor()
    except FileNotFoundError:
        return None


@bp.route("/")
def index():
    predictor = _get_predictor()
    metrics = predictor.get_metrics() if predictor else {}
    return render_template("index.html", metrics=metrics)


@bp.route("/form")
def form():
    return render_template("form.html")


@bp.route("/predict", methods=["POST"])
def predict():
    predictor = _get_predictor()
    if not predictor:
        flash("Model belum dilatih. Jalankan `python -m ml.train` dulu.", "danger")
        return redirect(url_for("main.form"))

    # Cabang 1: upload CSV
    file = request.files.get("uji_csv")
    if file and file.filename:
        try:
            df = pd.read_csv(file)
            result = predictor.evaluate_csv(df)
        except ValueError as e:
            flash(str(e), "danger")
            return redirect(url_for("main.form"))
        except Exception:
            flash("File CSV tidak valid atau rusak.", "danger")
            return redirect(url_for("main.form"))

        return render_template(
            "hasil.html",
            mode="csv",
            hasil="Evaluasi dataset CSV berhasil.",
            data={},
            tabel_jarak=[],
            **result,
        )

    # Cabang 2: input manual
    data, errors = validate_form_input(request.form)
    if errors:
        for err in errors:
            flash(err, "danger")
        return redirect(url_for("main.form"))

    features = form_to_features(data)
    pred, label, tabel = predictor.predict_single(features)

    return render_template(
        "hasil.html",
        mode="manual",
        hasil=label,
        data=data,
        tabel_jarak=tabel,
        akurasi=0,
        presisi=0,
        recall=0,
        jumlah_benar=0,
        jumlah_salah=0,
        cm_dict={"TP": "-", "FN": "-", "FP": "-", "TN": "-"},
        n_test=0,
    )


@bp.route("/metrics")
def metrics_page():
    predictor = _get_predictor()
    if not predictor:
        flash("Model belum dilatih.", "warning")
        return redirect(url_for("main.index"))
    return render_template("metrics.html", metrics=predictor.get_metrics())


@bp.route("/tentang")
def tentang():
    return render_template("tentang.html")