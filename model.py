import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

class DiabetesPredictor:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=7)  # Increased neighbors for better generalization
        self.scaler = StandardScaler()

    def train(self, data):
        """Train the KNN model"""
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def predict(self, features):
        """Make prediction for given features"""
        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features_array)

        # Get prediction and probability
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        # Return prediction (0 or 1) and probability of the predicted class
        probability = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
        return prediction[0], probability