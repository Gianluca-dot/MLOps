import streamlit as st
import json
import os

st.set_page_config(page_title="MLOps Sentiments Monitoring", page_icon="📊", layout="wide")

st.title("📊 MLOps Sentiments Monitoring Dashboard")
st.markdown("Monitoraggio in tempo reale delle performance del modello di Sentiment Analysis.")

# Caricamento metriche
metrics_path = "data/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Accuracy", value=f"{metrics.get('accuracy', 0):.4f}")
    col2.metric(label="F1 Macro", value=f"{metrics.get('f1_macro', 0):.4f}")
    col3.metric(label="F1 Weighted", value=f"{metrics.get('f1_weighted', 0):.4f}")
else:
    st.warning("⚠️ File delle metriche (`data/metrics.json`) non ancora trovato. Esegui il training per generarlo.")

st.sidebar.header("Informazioni Pipeline")
st.sidebar.info("Repository: MLOps_Sentiments_Monitoring\nModello: cardiffnlp/twitter-roberta-base-sentiment-latest")
