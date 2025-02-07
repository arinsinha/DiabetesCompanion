import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_k = None

    def find_optimal_k(self, X_train, X_test, y_train, y_test):
        """Find the optimal K value by testing accuracy for different K values"""
        k_values = range(1, 21)
        accuracies = []

        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        best_k = k_values[np.argmax(accuracies)]
        best_accuracy = max(accuracies)

        return best_k, best_accuracy, accuracies

    def train(self, data):
        """Train the KNN model with optimal K value"""
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

        # Find optimal K
        self.best_k, best_accuracy, accuracies = self.find_optimal_k(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Train final model with best K
        self.model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.model.fit(X_train_scaled, y_train)

        return best_accuracy, accuracies, self.best_k

    def predict(self, features):
        """Make prediction for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

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