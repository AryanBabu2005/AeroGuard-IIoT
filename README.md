# ✈️ AeroGuard IIoT: Real-Time Predictive Maintenance & Digital Twin

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AeroGuard IIoT** is a production-grade predictive maintenance pipeline and digital twin control interface. It ingests high-frequency turbofan engine telemetry to forecast **Remaining Useful Life (RUL)** using a decoupled machine learning microservice architecture.

Trained on the **NASA C-MAPSS** run-to-failure dataset, the system transitions industrial maintenance from reactive scheduling to proactive, condition-based interventions based on stochastic sensor volatility.

---

## 🏗️ System Architecture

The platform is decoupled into two Dockerized microservices communicating via an internal bridge network:

```mermaid
graph LR
    A[Telemetry Stream] -->|15 Raw Sensors| B(FastAPI Inference Engine)
    B -->|Z-Score Normalization| C{Feature Expansion}
    C -->|30 Rolling Stats| D[Random Forest Ensemble]
    D -->|RUL & Uncertainty| E(Streamlit Mission Control)
    E -->|SHAP Diagnostics| F[Maintenance Engineer]