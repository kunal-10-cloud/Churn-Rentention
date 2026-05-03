import streamlit as st
import pickle
import json
import os

@st.cache_resource
def load_model():
    path = "models/model.pkl"
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    path = "models/scaler.pkl"
    if not os.path.exists(path):
        st.error(f"Scaler file not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_metrics():
    path = "models/metrics.json"
    if not os.path.exists(path):
        st.error(f"Metrics file not found at {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_feature_names():
    path = "models/feature_names.pkl"
    if not os.path.exists(path):
        st.error(f"Feature names file not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
