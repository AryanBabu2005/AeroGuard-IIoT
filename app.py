import streamlit as st
import joblib
import tensorflow as tf
from src.processor import add_rolling_features, create_sequences
from src.streamer import get_live_sensor_stream

# Sidebar Switcher
st.sidebar.header("🧠 Intelligence Layer")
model_type = st.sidebar.radio("Select Model Version", ["v1: Random Forest (Mid-Sem)", "v2: LSTM (Final-Sem)"])

@st.cache_resource
def load_selected_model(m_type):
    if m_type == "v1: Random Forest (Mid-Sem)":
        return joblib.load('models/rf_baseline.pkl')
    else:
        return tf.keras.models.load_model('models/lstm_advanced.h5')

model = load_selected_model(model_type)

# Prediction Logic inside your simulation loop
if model_type == "v1: Random Forest (Mid-Sem)":
    # Use standard 2D feature vector
    X_input = frame[feature_cols].values.reshape(1, -1)
    prediction = model.predict(X_input)[0]
else:
    # Use 3D sequence from buffer
    if len(streamed_buffer) >= 20:
        X_seq = create_sequences(pd.DataFrame(streamed_buffer).tail(20), feature_cols)
        prediction = model.predict(X_seq)[0][0]
    else:
        prediction = "Warming up..."