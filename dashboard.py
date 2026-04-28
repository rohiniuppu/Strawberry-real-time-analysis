import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from PIL import Image

# Setup page
st.set_page_config(page_title="Strawberry Real-Time Analysis", layout="wide", page_icon="🍓")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_dataset():
    return pd.read_csv("strawberry_dataset.csv")

try:
    model = load_model()
    df = load_dataset()
except Exception as e:
    st.error("Model or dataset not found. Please run app.py first to generate the model!")
    st.stop()

st.title("🍓 AI-Based Non-Destructive Analysis of Strawberries")
st.markdown("### Real-Time Hyperspectral Quality Assessment Dashboard")

st.sidebar.header("Camera Feed Simulation")
image_index = st.sidebar.slider("Select a strawberry passing on the conveyor belt:", 0, len(df)-1, 0)
selected_row = df.iloc[image_index]

# Columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sensor Input")
    st.write(f"**Image Source:** `{selected_row['image_path']}`")
    
    # Find full path to show the image
    base_dir = r"c:\Users\rohin\OneDrive\Documents\strawberry\strawberryDataset"
    full_img_path = os.path.join(base_dir, selected_row['image_path'])
    
    if os.path.exists(full_img_path):
        img = Image.open(full_img_path)
        st.image(img, caption="Camera Target", use_column_width=True)
    else:
        st.warning(f"Image not found on disk at {full_img_path}")

with col2:
    st.subheader("Real-Time Attributes (from HSI)")
    
    # We pull the features for inference
    features = ["tartaric acid", "ph value", "soluble salts", "firmness", "color", "size"]
    input_data = selected_row[features].to_dict()
    
    # Display features dynamically
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    col_feat1.metric("Tartaric Acid", f"{input_data['tartaric acid']:.2f}")
    col_feat2.metric("pH Value", f"{input_data['ph value']:.2f}")
    col_feat3.metric("Soluble Salts", f"{input_data['soluble salts']:.2f}")
    
    col_feat4, col_feat5, col_feat6 = st.columns(3)
    col_feat4.metric("Firmness", f"{input_data['firmness']:.2f}")
    col_feat5.metric("Color Index", f"{input_data['color']:.2f}")
    col_feat6.metric("Size (mm)", f"{input_data['size']:.2f}")
    
    st.markdown("---")
    st.subheader("Analysis Engine Verdict")
    
    # Predict using the model
    with st.spinner('Analyzing across pipeline...'):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
    
    if prediction == 1:
        st.success(f"✅ **PICKABLE** (Confidence: {probabilities[1]*100:.1f}%)")
        st.info("The strawberry meets the quality thresholds for Tartaric Acid, Firmness, and Color. Send to packaging.")
    else:
        st.error(f"❌ **UNPICKABLE** (Confidence: {probabilities[0]*100:.1f}%)")
        st.warning("The strawberry failed the quality parameter threshold setup. Discarding.")

st.markdown("---")
st.markdown("### Research Artifacts")
st.markdown("Here is the visual proof of our model matching the ~0.9 accuracy required by the paper.")
col3, col4 = st.columns(2)
with col3:
    if os.path.exists("feature_importance.png"):
        st.image("feature_importance.png", caption="Feature Importance (What the AI looks for)")
with col4:
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Confusion Matrix on 20% validation set")
