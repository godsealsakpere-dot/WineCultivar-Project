import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="Wine Cultivar Predictor", layout="centered")

# Title and Overview
st.title("🍷 Wine Cultivar Origin Prediction System")
st.write("""
This system predicts the origin (cultivar) of a wine based on its chemical properties.
Please input the chemical values below.
""")

# Load the saved model
# We use a try-except block to handle path issues depending on where the app is run
model_path = os.path.join('model', 'wine_cultivar_model.pkl')

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Fallback if running from the root and model is in the same directory
        model = joblib.load('wine_cultivar_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create Input Form
st.subheader("Enter Wine Chemical Properties")

# We create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", min_value=10.0, max_value=15.0, value=13.0, step=0.1)
    magnesium = st.number_input("Magnesium", min_value=70.0, max_value=170.0, value=100.0, step=1.0)
    flavanoids = st.number_input("Flavanoids", min_value=0.3, max_value=5.1, value=2.0, step=0.1)

with col2:
    color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1)
    hue = st.number_input("Hue", min_value=0.4, max_value=1.8, value=1.0, step=0.01)
    proline = st.number_input("Proline", min_value=200.0, max_value=1700.0, value=700.0, step=10.0)

# Prediction Button
if st.button("Predict Cultivar"):
    # Prepare input array (must match the order used in training)
    # Features: ['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline']
    input_data = np.array([[alcohol, magnesium, flavanoids, color_intensity, hue, proline]])
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction to Cultivar Name (0, 1, 2 correspond to Cultivar 1, 2, 3)
    cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
    result = cultivar_map.get(prediction, "Unknown")
    
    # Display Result
    st.success(f"The predicted origin is: **{result}**")
    
    # Optional: Display probability/confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)
        st.info(f"Confidence: {np.max(proba)*100:.2f}%")