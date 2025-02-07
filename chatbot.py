import google.generativeai as genai
import streamlit as st

class DiabetesChatbot:
    def __init__(self):
        # Check if API key is configured
        if "GEMINI_API_KEY" not in st.secrets:
            raise ValueError("Gemini API key not found in secrets")

        # Initialize Gemini API
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        # Set up the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Initialize chat history
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_input):
        """Get response from Gemini for user query"""
        try:
            context = """You are a medical assistant specializing in diabetes. 
            Provide accurate, helpful information about diabetes, its prevention, 
            and management. Keep responses concise and easy to understand."""

            prompt = f"{context}\nUser: {user_input}\nAssistant:"
            response = self.chat.send_message(prompt)
            return response.text

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"