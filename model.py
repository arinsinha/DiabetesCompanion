import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

class DiabetesPredictor:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
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
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict(self, features):
        """Make prediction for given features"""
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)
        
        return prediction[0], probability[0][1]
