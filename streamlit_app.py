import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Churn Insight Platform", page_icon="🏦", layout="wide")

st.title("🏦 Churn Insight: Predicción de Abandono")
st.markdown("Herramienta de análisis de riesgo para clientes bancarios.")
st.divider()

MODEL_PATH = "modelo_Banco_churn.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error al cargar modelo: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error(f"❌ Archivo no encontrado: {MODEL_PATH}")
        st.write("Archivos disponibles:", os.listdir("."))
    return None

pipe_xgb = load_model()

if pipe_xgb is not None:
    with st.sidebar:
        st.header("📋 Datos del Cliente")
        st.markdown("---")

        st.subheader("👤 Perfil")
