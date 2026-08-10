# Churn Prediction Web App

This is a Flask-based web application that predicts customer churn based on user-provided inputs. The model used is a Logistic Regression trained on telecom churn data.

## Features
- User registration and login
- Input customer features to get churn prediction
- Trained model with pre-processing (scaler and feature list)
- Static UI with HTML/CSS

## Tech Stack
- Python, Flask
- Scikit-learn (model training - pre-bundled)
- HTML/CSS for UI

## How to Run Locally
```bash
pip install -r requirements.txt
python churn_app/app.py
