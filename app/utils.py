"""Helper validasi form."""
from config import Config


def validate_form_input(form) -> tuple[dict, list]:
    """Validasi input form manual. Return (data_bersih, list_error)."""
    errors = []
    data = {}

    ranges = {
        "pregnancies": (0, 20, "Jumlah kehamilan"),
        "glucose": (1, 300, "Kadar glukosa"),
        "bloodPressure": (1, 200, "Tekanan darah"),
        "skinThickness": (1, 100, "Ketebalan kulit"),
        "insulin": (1, 900, "Kadar insulin"),
        "bmi": (1, 70, "BMI"),
        "dpf": (0, 3, "Diabetes Pedigree Function"),
        "age": (1, 120, "Usia"),
    }

    for key, (lo, hi, label) in ranges.items():
        raw = (form.get(key) or "").strip()
        if raw == "":
            errors.append(f"{label} wajib diisi.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{label} harus antara {lo} dan {hi}.")
            continue
        data[key] = val

    return data, errors


FORM_TO_FEATURE = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bloodPressure": "BloodPressure",
    "skinThickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "dpf": "DiabetesPedigreeFunction",
    "age": "Age",
}


def form_to_features(data: dict) -> dict:
    return {FORM_TO_FEATURE[k]: v for k, v in data.items()}