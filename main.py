import streamlit as st
import pandas as pd
import plotly.express as px
from model import DiabetesPredictor
from chatbot import DiabetesChatbot
from utils import load_data, create_feature_plot, get_model_features, validate_input

# Page config
st.set_page_config(
    page_title="Diabetes Prediction Assistant",
    page_icon="🏥",
    layout="wide"
)

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = DiabetesChatbot()
if 'predictor' not in st.session_state:
    st.session_state.predictor = DiabetesPredictor()
    data = load_data()
    accuracy = st.session_state.predictor.train(data)

# Main header
st.title("🏥 Diabetes Prediction Assistant")
st.markdown("""
    This application helps predict diabetes risk using machine learning and provides
    expert information through an AI chatbot. Enter your health metrics below for a prediction.
""")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Prediction Model")
    
    # Input form
    with st.form("prediction_form"):
        features = {}
        for feature in get_model_features():
            features[feature] = st.number_input(
                f"Enter {feature}",
                min_value=0.0,
                help=f"Input value for {feature}"
            )
        
        submit_button = st.form_submit_button("Get Prediction")
        
        if submit_button:
            # Validate inputs
            values = list(features.values())
            is_valid, error_message = validate_input(values)
            
            if is_valid:
                # Make prediction
                prediction, probability = st.session_state.predictor.predict(values)
                
                # Display results
                st.markdown("### Results")
                if prediction == 1:
                    st.error(f"⚠️ High risk of diabetes (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ Low risk of diabetes (Probability: {1-probability:.2%})")
                
                st.info("""
                    Note: This is a preliminary screening tool. Please consult with a 
                    healthcare professional for proper medical advice and diagnosis.
                """)
            else:
                st.error(error_message)

with col2:
    st.header("💬 AI Health Assistant")
    st.markdown("""
        Ask any questions about diabetes, its prevention, 
        symptoms, or management.
    """)
    
    # Chat interface
    user_input = st.text_input("Your question:")
    if st.button("Ask"):
        if user_input:
            response = st.session_state.chatbot.get_response(user_input)
            st.markdown(f"**Response:**\n{response}")

# Data Visualization Section
st.header("📈 Data Insights")
data = load_data()

# Feature selection for visualization
selected_feature = st.selectbox(
    "Select feature to visualize:",
    get_model_features()
)

# Display distribution plot
st.plotly_chart(create_feature_plot(data, selected_feature), use_container_width=True)

# Footer
st.markdown("""
    ---
    Made with ❤️ by Your Name | Data source: PIMA Indians Diabetes Dataset
""")
