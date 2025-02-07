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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = DiabetesPredictor()
    data = load_data()
    accuracy, accuracies, best_k = st.session_state.predictor.train(data)
    st.session_state.model_metrics = {
        'accuracy': accuracy,
        'accuracies': accuracies,
        'best_k': best_k
    }

# Try to initialize chatbot
try:
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DiabetesChatbot()
    chatbot_available = True
except Exception as e:
    chatbot_available = False

# Main header with enhanced description
st.markdown("""
    <div class='main-header'>
        <h1>🏥 Diabetes Prediction Assistant</h1>
        <p style='font-size: 1.2rem; max-width: 800px; margin: 0 auto;'>
            Welcome to our advanced diabetes risk assessment platform. Using state-of-the-art machine learning 
            and artificial intelligence, we analyze your health metrics to provide personalized risk assessments 
            and expert insights. Our comprehensive system combines a sophisticated KNN-based prediction model 
            with an AI health assistant to offer detailed diabetes risk analysis, management guidance, and 
            preventive recommendations tailored to your health profile.
        </p>
    </div>
""", unsafe_allow_html=True)

# Risk Assessment Section
st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.header("📊 Risk Assessment Model")

# Input form
with st.form("prediction_form"):
    features = {}
    for feature in get_model_features():
        features[feature] = st.number_input(
            f"Enter {feature}",
            min_value=0.0
        )

    submit_button = st.form_submit_button("Get Prediction")

    if submit_button:
        values = list(features.values())
        is_valid, error_message = validate_input(values)

        if is_valid:
            prediction, probability = st.session_state.predictor.predict(values)

            st.markdown("### Results")
            if prediction == 1:
                risk_percentage = probability * 100
                st.error(f"⚠️ High risk of diabetes (Probability: {risk_percentage:.1f}%)")
            else:
                risk_percentage = (1 - probability) * 100
                st.success(f"✅ Low risk of diabetes (Probability: {risk_percentage:.1f}%)")

            st.info(f"""
                **Prediction Details:**
                - Risk Level: {"High" if prediction == 1 else "Low"}
                - Confidence: {max(probability, 1-probability):.1%}

                Note: This is a preliminary screening tool. Please consult with a 
                healthcare professional for proper medical advice and diagnosis.
            """)
        else:
            st.error(error_message)
st.markdown("</div>", unsafe_allow_html=True)

# AI Health Assistant Section
st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.header("🤖 AI Health Assistant")

if chatbot_available:
    st.markdown("""
        Our AI Health Assistant is powered by advanced natural language processing to provide 
        comprehensive information about diabetes. Get expert insights about prevention strategies, 
        daily management tips, dietary recommendations, exercise guidelines, and understanding 
        risk factors. Feel free to ask any questions about diabetes symptoms, treatment options, 
        lifestyle modifications, or general health concerns.
    """)

    user_input = st.text_input("Your question:", placeholder="e.g., What are the early signs of diabetes?")
    if st.button("Ask Assistant"):
        if user_input:
            with st.spinner("Getting response..."):
                response = st.session_state.chatbot.get_response(user_input)
                st.markdown(f"<div class='chat-response'>{response}</div>", unsafe_allow_html=True)
else:
    st.warning("""
        The AI Health Assistant is currently unavailable. 
        Please ask an administrator to configure the Gemini API key.
    """)
st.markdown("</div>", unsafe_allow_html=True)

# Model Performance Section
st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.header("🎯 Model Performance")
if 'model_metrics' in st.session_state:
    metrics = st.session_state.model_metrics

    metrics_cols = st.columns(2)
    with metrics_cols[0]:
        st.metric("Best K Value", metrics['best_k'])
    with metrics_cols[1]:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")

    k_values = range(1, 21)
    k_accuracy_df = pd.DataFrame({
        'K Value': k_values,
        'Accuracy': metrics['accuracies']
    })

    fig = px.line(k_accuracy_df, x='K Value', y='Accuracy',
                  title='Model Accuracy vs K Value',
                  markers=True)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#E0E0E0'},
        xaxis=dict(gridcolor='rgba(155, 89, 182, 0.1)'),
        yaxis=dict(gridcolor='rgba(155, 89, 182, 0.1)'),
    )
    fig.add_vline(x=metrics['best_k'], line_dash="dash",
                  annotation_text=f"Best K = {metrics['best_k']}")
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Data Visualization Section
st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.header("📈 Data Insights")
data = load_data()

selected_feature = st.selectbox(
    "Select feature to visualize:",
    get_model_features()
)

fig = create_feature_plot(data, selected_feature)
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font={'color': '#E0E0E0'},
    xaxis=dict(gridcolor='rgba(155, 89, 182, 0.1)'),
    yaxis=dict(gridcolor='rgba(155, 89, 182, 0.1)'),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <p>Made with ❤️ by Arin Ved Sinha | Data source: PIMA Indians Diabetes Dataset</p>
        <p style='font-size: 0.9rem;'>© 2024 Diabetes Prediction Assistant. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    st.run(app, host='0.0.0.0', port=8501)