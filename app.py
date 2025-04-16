import streamlit as st
import numpy as np
import random
import tensorflow as tf

st.set_page_config(page_title="Fire Detection", layout="centered")

st.title("🔥 Smart Fire Detection System")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="fire_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

FEATURES = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'PM2.5']

def predict_fire(features):
    input_data = np.array([features], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return int(output_data[0][0])

def get_random_sensor_data():
    return [
        round(random.uniform(20, 80), 2),    # Temperature[C]
        round(random.uniform(10, 90), 2),    # Humidity[%]
        round(random.uniform(0, 600), 2),    # TVOC[ppb]
        round(random.uniform(400, 2000), 2), # eCO2[ppm]
        round(random.uniform(0, 150), 2),    # PM2.5
    ]

tab1, tab2 = st.tabs(["📡 Live Sensor Feed", "🛠️ Custom Input (Advanced)"])

# --- Tab 1: IoT-style sensor data ---
with tab1:
    st.subheader("Fetch Data from IoT Sensor")
    if st.button("Fetch Sensor Data"):
        data = get_random_sensor_data()
        prediction = predict_fire(data)
        st.success("Sensor Data Received")
        st.write(
            {
                FEATURES[i]: data[i] for i in range(len(FEATURES))
            }
        )
        st.markdown(f"### 🔍 Fire Alarm Status: {'🚨 Fire Detected!' if prediction else '✅ Safe'}")

# --- Tab 2: Hidden Custom Input (Advanced) ---
with tab2:
    st.subheader("Enter Custom Values")
    custom_data = []
    for feature in FEATURES:
        val = st.number_input(f"{feature}", min_value=0.0, step=0.1)
        custom_data.append(val)
    
    if st.button("Predict with Custom Input"):
        result = predict_fire(custom_data)
        st.markdown(f"### 🔍 Fire Alarm Status: {'🚨 Fire Detected!' if result else '✅ Safe'}")

# Hide the custom input tab visually but keep it functional
hide_custom_css = """
    <style>
    button[data-baseweb="tab"]:nth-child(2) {
        opacity: 0.3;
    }
    </style>
"""
st.markdown(hide_custom_css, unsafe_allow_html=True)
