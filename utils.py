import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

def load_data():
    """Load and preprocess the diabetes dataset"""
    data = pd.read_csv('attached_assets/diabetes.csv')
    return data

def create_feature_plot(data, feature):
    """Create distribution plot for a given feature"""
    fig = px.histogram(data, x=feature, color='Outcome',
                      marginal='box',
                      title=f'Distribution of {feature} by Diabetes Outcome',
                      color_discrete_sequence=['#2E86C1', '#E74C3C'])
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#262730'}
    )
    return fig

def get_model_features():
    """Return list of features needed for prediction"""
    return [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

def validate_input(values):
    """Validate input values"""
    if any(v < 0 for v in values):
        return False, "All values must be non-negative"

    validations = {
        'Glucose': (0, 200),
        'BloodPressure': (0, 130),
        'BMI': (0, 70),
        'Age': (0, 120)
    }

    for i, feature in enumerate(get_model_features()):
        if feature in validations:
            min_val, max_val = validations[feature]
            if not min_val <= values[i] <= max_val:
                return False, f"{feature} must be between {min_val} and {max_val}"

    return True, ""