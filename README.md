# Diabetes Prediction Assistant

## Overview
DiabetesCompanion is a machine learning-powered web application designed to predict diabetes risk based on user inputs. The application utilizes the **Pima Indian Diabetes Dataset** to generate predictions and provide insights into potential diabetes risks.

## Features
- **User-friendly Interface**: Simple and intuitive UI for users to input health data.
- **Machine Learning Model**: Predicts the likelihood of diabetes using trained models.
- **Visualization**: Graphical representation of data and results.
- **Interactive Insights**: Provides recommendations based on model predictions.
- **Cloud Deployment**: Hosted using AWS services for scalability.

## Tech Stack
- **Frontend**: React.js
- **Backend**: Node.js, Express.js
- **Machine Learning**: Python (Scikit-learn, Pandas, NumPy, Seaborn)
- **Database**: MongoDB
- **Cloud Services**: AWS Amplify, AWS Lambda, Amazon Bedrock, AWS Cognito

## Installation
### Prerequisites
Ensure you have the following installed:
- Node.js
- Python 3.x
- MongoDB

### Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/arinsinha/DiabetesCompanion.git
   cd DiabetesCompanion
   ```
2. **Install dependencies**
   - Backend:
     ```sh
     cd backend
     pip install -r requirements.txt
     ```
   - Frontend:
     ```sh
     cd frontend
     npm install
     ```
3. **Run the application**
   - Start the backend:
     ```sh
     python app.py
     ```
   - Start the frontend:
     ```sh
     npm start
     ```

## Usage
1. Open the application in your browser.
2. Enter relevant health details such as glucose level, BMI, insulin level, etc.
3. Click the **Predict** button to get the results.
4. View predictions along with recommendations for diabetes prevention.

## Model Training
To train the machine learning model manually:
```sh
cd backend
python train_model.py
```

## Future Enhancements
- Integration with wearable devices for real-time health monitoring.
- Addition of more datasets for improved accuracy.
- Implementation of a chatbot for personalized health guidance.

## Contributing
Feel free to submit pull requests and open issues to improve the project.

## License
This project is licensed under the MIT License.

## Contact
For any queries or suggestions, reach out via [GitHub Issues](https://github.com/arinsinha/DiabetesCompanion/issues).

