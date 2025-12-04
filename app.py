import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# -------------------------------
# Load Pre-Trained Models
# -------------------------------
from pathlib import Path
import joblib
import json

models_dir = Path("models")

clf = joblib.load(models_dir / "iaq_clf.joblib")
rgr = joblib.load(models_dir / "iaq_rgr.joblib")
scaler = joblib.load(models_dir / "iaq_scaler.joblib")

# Load metadata.json correctly (use json.load, not joblib)
meta_path = models_dir / "metadata.json"
meta = json.load(open(meta_path)) if meta_path.exists() else None


st.set_page_config(layout="wide", page_title="Indoor Air Quality Monitoring System")

st.title("🌿 Indoor Air Quality Monitoring System using Machine Learning")
st.write("""
Upload indoor air-quality sensor data to automatically classify
air quality into **Good**, **Moderate**, or **Poor**, compute IAQ scores,
predict AQ values, and visualize trends.
""")

# -------------------------------
# Helper Functions
# -------------------------------

def compute_iaq_score(df):
    arr = df[['air_quality','dust_density','temperature','humidity']].astype(float).values
    scaled = scaler.transform(arr)
    df['iaq_score'] = (
        0.55 * scaled[:,0] +
        0.35 * scaled[:,1] +
        0.05 * scaled[:,2] +
        0.05 * scaled[:,3]
    ) * 100

    df['iaq_score'] = df['iaq_score'].clip(0,100).round(1)
    return df

def label_iaq(df):
    q1 = meta["quantiles"]["q1"]
    q2 = meta["quantiles"]["q2"]

    def f(x):
        if x <= q1: return "Good"
        elif x <= q2: return "Moderate"
        return "Poor"

    df['iaq_label'] = df['iaq_score'].apply(f)
    return df

# -------------------------------
# File Upload
# -------------------------------

uploaded = st.file_uploader("📤 Upload IAQ CSV file", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)

    # Fix column names
    df = df.rename(columns=lambda c: c.strip().lower())
    if "air quality" in df.columns:
        df = df.rename(columns={"air quality": "air_quality"})
    if "dust density" in df.columns:
        df = df.rename(columns={"dust density": "dust_density"})

    required_cols = ["temperature","humidity","air_quality","dust_density"]

    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain: {required_cols}")
        st.stop()

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    df = df.dropna(subset=required_cols)
    df[required_cols] = df[required_cols].astype(float)

    # Compute IAQ Score + Label + Predictions
    df = compute_iaq_score(df)
    df = label_iaq(df)

    # Classifier Prediction
    X = df[['temperature','humidity','dust_density']]
    df['predicted_label'] = clf.predict(X)

    # Regressor Prediction
    df['predicted_air_quality'] = rgr.predict(X).round(2)

    # -------------------------------
    # Show Results
    # -------------------------------

    st.subheader("🏷 Predicted IAQ Labels (first 20 rows)")
    st.dataframe(df[['temperature','humidity','dust_density','iaq_score','predicted_label']].head(20))

    # -------------------------------
    # Graphs
    # -------------------------------

    st.subheader("📉 IAQ Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["iaq_score"], bins=20, color="green")
    ax.set_xlabel("IAQ Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("🌫 Dust Density Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["dust_density"], bins=20, color="orange")
    ax.set_xlabel("Dust Density")
    st.pyplot(fig)

    # -------------------------------
    # Download Output
    # -------------------------------

    st.download_button(
        "⬇ Download Predictions",
        df.to_csv(index=False),
        "iaq_predictions.csv"
    )

# -------------------------------
# Manual Input Section
# -------------------------------

st.header("🧪 Manual IAQ Prediction")

col1, col2, col3, col4 = st.columns(4)
temp = col1.number_input("Temperature (°C)", 20.0)
hum = col2.number_input("Humidity (%)", 60.0)
dust = col3.number_input("Dust Density (PM)", 250.0)
aq = col4.number_input("Air Quality Value (sensor)", 450.0)

if st.button("Predict Now"):
    single = pd.DataFrame([{
        "temperature": temp,
        "humidity": hum,
        "dust_density": dust,
        "air_quality": aq
    }])

    single = compute_iaq_score(single)
    single = label_iaq(single)

    pred_label = clf.predict(single[['temperature','humidity','dust_density']])[0]
    pred_aq = rgr.predict(single[['temperature','humidity','dust_density']])[0]

    st.subheader(f"IAQ Label: **{pred_label}**")
    st.subheader(f"IAQ Score: **{single['iaq_score'].iloc[0]} / 100**")
    st.subheader(f"Predicted AQ Value: **{pred_aq:.2f}**")

    if pred_label == "Poor":
        st.error("⚠ Poor air quality. Improve ventilation, use purifier.")
    elif pred_label == "Moderate":
        st.warning("⚠ Moderate air quality. Monitor conditions.")
    else:
        st.success("✓ Good air quality.")

