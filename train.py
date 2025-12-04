# train.py
"""
Train IAQ classifier + regressor, save models and metadata.
Run: python train.py --input data/feed.csv --out models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_iaq_score(df, scaler=None, fit_scaler=True):
    """Create scaled features and IAQ score (0-100). Returns df, scaler."""
    features = ['air_quality', 'dust_density', 'temperature', 'humidity']
    # Ensure columns exist
    for c in features:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    # Keep a copy of raw features
    arr = df[features].astype(float).values
    if scaler is None and fit_scaler:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(arr)
    elif scaler is not None:
        scaled = scaler.transform(arr)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(arr)

    df[['aq_s','dust_s','temp_s','hum_s']] = scaled
    # Weighted IAQ score (pollutants heavier)
    df['iaq_score'] = (0.55*df['aq_s'] + 0.35*df['dust_s'] + 0.05*df['temp_s'] + 0.05*df['hum_s']) * 100
    df['iaq_score'] = df['iaq_score'].clip(0,100).round(1)
    return df, scaler

def make_labels_by_quantiles(df):
    q1 = float(df['iaq_score'].quantile(0.33))
    q2 = float(df['iaq_score'].quantile(0.66))
    def label(s):
        if s <= q1:
            return 'Good'
        elif s <= q2:
            return 'Moderate'
        else:
            return 'Poor'
    df['iaq_label'] = df['iaq_score'].apply(label)
    return df, {'q1': q1, 'q2': q2}

def train_models(df, out_dir):
    # Use features for modeling
    features = ['temperature','humidity','dust_density']
    X = df[features].astype(float)
    y_cls = df['iaq_label']
    y_reg = df['air_quality'].astype(float)

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train_cls)

    rgr = RandomForestRegressor(n_estimators=200, random_state=42)
    rgr.fit(X_train, y_train_reg)

    # Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "iaq_clf.joblib")
    joblib.dump(rgr, out_dir / "iaq_rgr.joblib")
    print(f"Saved models to {out_dir}")

    return clf, rgr

def main(args):
    data_path = Path(args.input)
    out_dir = Path(args.out)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    df = pd.read_csv(data_path)
    # Standardize column names if needed
    df = df.rename(columns=lambda c: c.strip().lower())
    # normalize likely column names
    rename_map = {}
    if 'air quality' in df.columns:
        rename_map['air quality'] = 'air_quality'
    if 'dust density' in df.columns:
        rename_map['dust density'] = 'dust_density'
    df = df.rename(columns=rename_map)

    # Ensure numeric types
    for c in ['temperature','humidity','air_quality','dust_density']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with NaNs
    df = df.dropna(subset=['temperature','humidity','air_quality','dust_density']).reset_index(drop=True)

    # Create IAQ score + scaler
    df, scaler = create_iaq_score(df, scaler=None, fit_scaler=True)
    df, quantiles = make_labels_by_quantiles(df)

    # Train models
    clf, rgr = train_models(df, out_dir)

    # Save scaler + quantiles metadata
    joblib.dump(scaler, out_dir / "iaq_scaler.joblib")
    meta = {
        "quantiles": quantiles,
        "features": ['temperature','humidity','dust_density'],
        "notes": "iaq_score = weighted combination of scaled features (see train.py)"
    }
    with open(out_dir / "metadata.json","w") as f:
        json.dump(meta, f, indent=2)

    # Save a small sample csv for quick demo
    df.sample(min(50, len(df))).to_csv(out_dir / "sample_data.csv", index=False)
    print("Training finished. Artifacts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/feed.csv", help="Path to input CSV")
    parser.add_argument("--out", type=str, default="models", help="Output dir for models")
    args = parser.parse_args()
    main(args)
