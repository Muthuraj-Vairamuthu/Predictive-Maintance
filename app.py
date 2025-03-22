import streamlit as st
import numpy as np
import pickle

# --- Page Setup ---
st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️")
st.title("🛠️ Predictive Maintenance Dashboard")
st.markdown("Enter sensor readings and predict machine failure risk.")

# --- Load Models and Tools ---
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("device_encoder.pkl", "rb") as f:
    device_encoder = pickle.load(f)

# --- Model Selector ---
model_choice = st.selectbox("Choose a Model:", ["XGBoost", "Random Forest", "Logistic Regression"])
model = {"XGBoost": xgb_model, "Random Forest": rf_model, "Logistic Regression": logreg_model}[model_choice]

# --- Inputs ---
st.subheader("Sensor Metrics:")
metrics = [st.slider(f"Metric {i+1}", -3.0, 3.0, 0.0, step=0.1) for i in range(9)]

device_id = st.text_input("Enter Device ID (e.g., S1F01085)")

# --- Predict ---
if device_id:
    try:
        device_encoded = device_encoder.transform([device_id])[0]
        input_data = np.array([*metrics, device_encoded]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.metric("Failure Probability", f"{probability:.4f}")
        if prediction == 1:
            st.error(" MACHINE FAILURE PREDICTED")
        else:
            st.success(" Machine is Safe")

    except Exception as e:
        st.error("Invalid Device ID. Please enter a known ID from the dataset.")
