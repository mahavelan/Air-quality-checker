# utils.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def load_models(models_dir="models"):
    models_dir = Path(models_dir)
    clf = joblib.load(models_dir / "iaq_clf.joblib")
    rgr = joblib.load(models_dir / "iaq_rgr.joblib")
    scaler = joblib.load(models_dir / "iaq_scaler.joblib")
    meta = json.load(open(models_dir / "metadata.json", "r"))
    return clf, rgr, scaler, meta

def compute_iaq_score_row(row, scaler):
    # expects row with air_quality,dust_density,temperature,humidity
    arr = np.array([[row['air_quality'], row['dust_density'], row['temperature'], row['humidity']]])
    scaled = scaler.transform(arr)[0]
    aq_s, dust_s, temp_s, hum_s = scaled
    score = (0.55*aq_s + 0.35*dust_s + 0.05*temp_s + 0.05*hum_s) * 100
    return float(np.clip(round(score,1), 0, 100))

def label_from_score(score, quantiles):
    q1 = quantiles['q1']; q2 = quantiles['q2']
    if score <= q1: return "Good"
    if score <= q2: return "Moderate"
    return "Poor"
