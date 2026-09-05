import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import shap
import warnings
import plotly.express as px
import os
import requests

# Get API URL from Docker environment, or default to localhost
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/v1/predict"

# Example usage inside your Streamlit logic:
# response = requests.post(PREDICT_ENDPOINT, json=payload)

from src.data_loader import load_data
from src.feature_eng import add_rolling_features
from src.streamer import get_live_sensor_stream

warnings.filterwarnings("ignore")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AeroGuard Engine Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CLEAN UI --------------------
st.markdown("""
<style>
.main {
    background-color: #0b0f14;
    color: #e6edf3;
}
section[data-testid="stSidebar"] {
    background-color: #11161c;
}
[data-testid="stMetric"] {
    background: #141a21;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2630;
}
div[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 600;
    color: #d1d9e0;
}
div[data-testid="stMetricLabel"] {
    color: #8b98a5;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: 500;
    border: none;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_assets():
    model = joblib.load('engine_model.pkl')
    raw_data = load_data('train_FD001.txt')
    processed_data = add_rolling_features(raw_data)
    features = [col for col in processed_data.columns if '_mean' in col or '_std' in col]
    explainer = shap.TreeExplainer(model)
    return model, processed_data, explainer, features

model, processed_data, explainer, feature_cols = load_assets()

# -------------------- SIDEBAR --------------------
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

engine_id = st.sidebar.selectbox(
    "Select Engine",
    processed_data['unit_nr'].unique()
)

sim_speed = st.sidebar.slider(
    "Update Speed (sec/frame)",
    0.1, 1.0, 0.3
)

st.sidebar.markdown("---")
start_btn = st.sidebar.button("Start Monitoring", use_container_width=True)

# -------------------- HEADER --------------------
st.title("AeroGuard Engine Monitor")
st.caption("Real-time health insights and predictive diagnostics")

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs([
    "Live Monitoring",
    "Signal Insights",
    "AI Diagnostics"
])

# -------------------- TAB 1 --------------------
with tab1:
    kpi_zone = st.empty()
    chart_zone = st.empty()
    log_zone = st.empty()

# -------------------- TAB 2 --------------------
with tab2:
    st.header("Signal Behavior Insights")
    st.caption("Understanding drift and variability in engine sensors")
    stats_zone = st.empty()

# -------------------- TAB 3 --------------------
with tab3:
    st.header("AI Diagnostics")
    st.caption("Key factors influencing engine health predictions")
    xai_zone = st.empty()

# -------------------- MAIN LOOP --------------------
if start_btn:

    streamed_buffer = []

    for frame in get_live_sensor_stream(processed_data, engine_id):

        time.sleep(sim_speed)
        streamed_buffer.append(frame)

        recent_df = pd.DataFrame(streamed_buffer)

        # Prepare input
        X_input = frame[feature_cols].to_frame().T
        current_rul = model.predict(X_input)[0]

        preds = [tree.predict(X_input)[0] for tree in model.estimators_]
        confidence = np.std(preds)

        # ---------------- KPI ----------------
        with kpi_zone.container():
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Remaining Life", f"{int(current_rul)} cycles")
            c2.metric("Current Cycle", int(frame['time_cycles']))

            c3.metric(
                "Prediction Spread",
                f"±{confidence:.1f}",
                delta="High variance" if confidence > 15 else "Stable"
            )

            status = "Healthy" if current_rul > 50 else "Needs Attention"

            if status == "Healthy":
                c4.success("System Status: Healthy")
            else:
                c4.warning("System Status: Needs Attention")

        # ---------------- CHARTS ----------------
        with chart_zone.container():

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Thermal Behavior**")
                st.line_chart(
                    recent_df.set_index('time_cycles')[['s_11_mean']],
                    height=260
                )

            with col2:
                st.markdown("**Pressure Stability**")
                st.area_chart(
                    recent_df.set_index('time_cycles')[['s_4_std']],
                    height=260
                )

        # ---------------- STATS ----------------
        with stats_zone.container():

            st.subheader("Sensor Variability")

            s1, s2 = st.columns(2)

            with s1:
                st.markdown("**Mean Drift (s_12_mean)**")
                st.line_chart(
                    recent_df.set_index('time_cycles')[['s_12_mean']]
                )

            with s2:
                st.markdown("**Noise Level (s_12_std)**")
                st.area_chart(
                    recent_df.set_index('time_cycles')[['s_12_std']]
                )

            st.caption("Rising variability often indicates mechanical stress buildup.")

        # ---------------- XAI (FIXED) ----------------
        with xai_zone.container():

            shap_values = explainer.shap_values(X_input)

            impact_df = pd.DataFrame({
                'Feature': feature_cols,
                'Impact': shap_values[0]
            }).sort_values(by='Impact', key=abs, ascending=False).head(8)

            st.subheader(f"Key Drivers at Cycle {int(frame['time_cycles'])}")

            # 🔥 Better Plot (Industry level)
            fig = px.bar(
                impact_df,
                x="Impact",
                y="Feature",
                orientation="h"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Context message
            if current_rul < 80:
                st.caption("These factors are actively contributing to degradation.")
            else:
                st.caption("These are the most influential features under normal operation.")

        # ---------------- ALERT ----------------
        if current_rul < 45:
            log_zone.warning(
                f"Cycle {int(frame['time_cycles'])}: Early signs of degradation detected. Monitoring recommended."
            )

        # ---------------- TOAST ----------------
        if current_rul < 45 and int(frame['time_cycles']) % 5 == 0:
            st.toast("Engine health dropping. Check diagnostics.", icon="⚠️")