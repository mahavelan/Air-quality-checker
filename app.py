# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from utils import load_models, compute_iaq_score_row, label_from_score

st.set_page_config(layout="wide", page_title="IAQ Analyzer")
st.title("Indoor Air Quality (IAQ) Monitoring & Prediction")

models_dir = Path("models")
if not models_dir.exists():
    st.warning("Models not found. Run `python train.py --input data/feed.csv --out models` first or upload models into /models.")
    
# Load models if available
try:
    clf, rgr, scaler, meta = load_models("models")
except Exception as e:
    clf = rgr = scaler = None
    meta = None

st.markdown("**Instructions:** Upload a CSV with columns: `temperature`, `humidity`, `air_quality`, `dust_density` OR use the manual entry to predict a single observation.")

# Sidebar - manual input
st.sidebar.header("Manual input for a single reading")
temp = st.sidebar.number_input("Temperature (°C)", value=30.0, format="%.1f")
hum = st.sidebar.number_input("Humidity (%)", value=69.0, format="%.1f")
dust = st.sidebar.number_input("Dust density (PM)", value=250.0, format="%.2f")
aq = st.sidebar.number_input("Air quality (raw sensor value, optional)", value=600.0, format="%.1f")

if st.sidebar.button("Predict (manual)"):
    if clf is None:
        st.error("Models not loaded. Train and save models first.")
    else:
        X = pd.DataFrame([[temp, hum, dust]], columns=['temperature','humidity','dust_density'])
        pred_label = clf.predict(X)[0]
        try:
            score = compute_iaq_score_row({'air_quality':aq,'dust_density':dust,'temperature':temp,'humidity':hum}, scaler)
        except Exception:
            score = None
        st.sidebar.markdown(f"### IAQ: **{pred_label}**")
        if score is not None:
            st.sidebar.markdown(f"**IAQ Score:** {score}/100")
        st.sidebar.write("**Recommendations:**")
        if pred_label == 'Poor':
            st.sidebar.write("- Increase ventilation\n- Use air purifier\n- Reduce dust sources")
        elif pred_label == 'Moderate':
            st.sidebar.write("- Monitor and improve ventilation")
        else:
            st.sidebar.write("- Conditions look fine")

# File upload for bulk analysis
st.header("Upload dataset for batch analysis")
uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df = df.rename(columns=lambda c: c.strip().lower())
    # normalize names
    if 'air quality' in df.columns:
        df = df.rename(columns={'air quality':'air_quality'})
    if 'dust density' in df.columns:
        df = df.rename(columns={'dust density':'dust_density'})
    st.subheader("Raw data (first rows)")
    st.dataframe(df.head())

    # basic validation
    needed = ['temperature','humidity','air_quality','dust_density']
    if not all(c in df.columns for c in needed):
        st.error(f"Uploaded CSV must contain columns: {needed}")
    else:
        df = df.dropna(subset=needed)
        # Compute IAQ score if scaler available
        if scaler is not None:
            # compute scores vectorized
            arr = df[['air_quality','dust_density','temperature','humidity']].astype(float).values
            scaled = scaler.transform(arr)
            df['iaq_score'] = (0.55*scaled[:,0] + 0.35*scaled[:,1] + 0.05*scaled[:,2] + 0.05*scaled[:,3]) * 100
            df['iaq_score'] = df['iaq_score'].clip(0,100).round(1)
            q1 = meta['quantiles']['q1']; q2 = meta['quantiles']['q2']
            df['iaq_label'] = df['iaq_score'].apply(lambda s: 'Good' if s<=q1 else ('Moderate' if s<=q2 else 'Poor'))
        # Show summary
        st.subheader("Summary statistics")
        st.write(df[['temperature','humidity','air_quality','dust_density']].describe())
        # Plot single pollutant time distribution if timestamp exists
        if 'created_at' in df.columns or 'timestamp' in df.columns:
            time_col = 'created_at' if 'created_at' in df.columns else 'timestamp'
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                st.subheader("Time-series plots")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(df[time_col], df['iaq_score'], marker='.', linestyle='-', linewidth=0.6)
                    ax.set_title("IAQ Score over time")
                    ax.set_xlabel("Time")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(df[time_col], df['dust_density'], marker='.', linestyle='-', linewidth=0.6)
                    ax.set_title("Dust density over time")
                    ax.set_xlabel("Time")
                    st.pyplot(fig)
            except Exception:
                st.info("Could not parse timestamp column for time-series plotting.")
        else:
            st.info("No timestamp column found — upload time column (created_at/timestamp) for trends.")

        # Predict labels using classifier
        if clf is not None:
            Xu = df[['temperature','humidity','dust_density']].astype(float)
            df['pred_label'] = clf.predict(Xu)
            st.subheader("Predicted IAQ labels (first 20 rows)")
            st.dataframe(df[['temperature','humidity','dust_density','iaq_score','pred_label']].head(20))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name="iaq_predictions.csv")

st.markdown("---")
st.write("Built with ❤️ — Nova")
