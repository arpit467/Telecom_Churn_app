from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os
import json
import warnings
from werkzeug.security import generate_password_hash, check_password_hash

# Suppress scikit-learn version warnings on pickle load
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = "churn_secret_key_123"

# Base directory path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and feature order safely
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURES_PATH)

# File to store user credentials
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ======================= ROUTES =======================

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        if username in users:
            error = "Username already exists."
        else:
            users[username] = generate_password_hash(password)
            save_users(users)
            return redirect(url_for("login"))

    return render_template("register.html", error=error)

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        user_pass = users.get(username)

        valid = False
        if user_pass:
            if user_pass.startswith("pbkdf2:") or user_pass.startswith("scrypt:"):
                valid = check_password_hash(user_pass, password)
            else:
                valid = (user_pass == password)
                if valid:
                    users[username] = generate_password_hash(password)
                    save_users(users)

        if valid:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)

@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username", ""))

@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    input_data = {col: 0 for col in feature_order}
    input_data.update({
        'SeniorCitizen': int(request.form["senior"]),
        'tenure': float(request.form["tenure"]),
        'MonthlyCharges': float(request.form["monthly"]),
        'TotalCharges': float(request.form["total"]),
        'gender_Male': 1 if request.form["gender"] == "Male" else 0,
        'Partner_Yes': 1 if request.form["partner"] == "Yes" else 0,
        'Dependents_Yes': 1 if request.form["dependents"] == "Yes" else 0,
        'PhoneService_Yes': 1 if request.form["phone"] == "Yes" else 0,
        'MultipleLines_No phone service': 1 if request.form["multiple"] == "No phone service" else 0,
        'MultipleLines_Yes': 1 if request.form["multiple"] == "Yes" else 0,
        'InternetService_Fiber optic': 1 if request.form["internet"] == "Fiber optic" else 0,
        'InternetService_No': 1 if request.form["internet"] == "No" else 0,
        'OnlineSecurity_No internet service': 1 if request.form["security"] == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if request.form["security"] == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if request.form["backup"] == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if request.form["backup"] == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if request.form["protection"] == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if request.form["protection"] == "Yes" else 0,
        'TechSupport_No internet service': 1 if request.form["support"] == "No internet service" else 0,
        'TechSupport_Yes': 1 if request.form["support"] == "Yes" else 0,
        'StreamingTV_No internet service': 1 if request.form["tv"] == "No internet service" else 0,
        'StreamingTV_Yes': 1 if request.form["tv"] == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if request.form["movies"] == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if request.form["movies"] == "Yes" else 0,
        'Contract_One year': 1 if request.form["contract"] == "One year" else 0,
        'Contract_Two year': 1 if request.form["contract"] == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if request.form["paperless"] == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if request.form["payment"] == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if request.form["payment"] == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if request.form["payment"] == "Mailed check" else 0
    })

    df = pd.DataFrame([input_data])[feature_order]
    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(df[["tenure", "MonthlyCharges", "TotalCharges"]])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    result = f"⚠️ Likely to Churn (Confidence: {prob:.2f})" if pred == 1 else f"✅ Not Likely to Churn (Confidence: {1 - prob:.2f})"
    return render_template("index.html", prediction=result, username=session.get("username", ""))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    return redirect(url_for("login"))

# ======================= RUN =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

