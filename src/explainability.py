import shap
import matplotlib.pyplot as plt
import streamlit as st

def explain_prediction(model, input_data, feature_names):
    """
    Uses SHAP values to explain WHY the AI predicted a specific RUL.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        input_data.iloc[0], 
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    return fig