from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# FINAL FEATURE ORDER (must match training)
columns = [
    'SeniorCitizen','tenure_yeo_trim','MonthlyCharges_qt_trim',
    'Partner','Dependents','Contract','PaperlessBilling',
    'gender_Male','PhoneService_Yes','MultipleLines_Yes',
    'InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_Yes','OnlineBackup_Yes','DeviceProtection_Yes',
    'TechSupport_Yes','StreamingTV_Yes','StreamingMovies_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    try:
        sim = request.form.get('sim')

        # SAFE INPUT HANDLING (NO CRASHES)
        data = {
            "SeniorCitizen": int(request.form.get('senior', 0)),
            "tenure_yeo_trim": float(request.form.get('tenure', 0)),
            "MonthlyCharges_qt_trim": float(request.form.get('monthly', 0)),

            "Partner": 1 if request.form.get('partner') == "Yes" else 0,
            "Dependents": 1 if request.form.get('dependents') == "Yes" else 0,

            "Contract": 1 if request.form.get('contract') == "One year"
                        else (2 if request.form.get('contract') == "Two year" else 0),

            "PaperlessBilling": 1 if request.form.get('paper') == "Yes" else 0,

            "gender_Male": 1 if request.form.get('gender') == "Male" else 0,
            "PhoneService_Yes": 1 if request.form.get('phone') == "Yes" else 0,
            "MultipleLines_Yes": 1 if request.form.get('multiple') == "Yes" else 0,

            "InternetService_Fiber optic": 1 if request.form.get('internet') == "Fiber optic" else 0,
            "InternetService_No": 1 if request.form.get('internet') == "No" else 0,

            "OnlineSecurity_Yes": 1 if request.form.get('security') == "Yes" else 0,
            "OnlineBackup_Yes": 1 if request.form.get('backup') == "Yes" else 0,
            "DeviceProtection_Yes": 1 if request.form.get('device') == "Yes" else 0,
            "TechSupport_Yes": 1 if request.form.get('support') == "Yes" else 0,
            "StreamingTV_Yes": 1 if request.form.get('tv') == "Yes" else 0,
            "StreamingMovies_Yes": 1 if request.form.get('movies') == "Yes" else 0,

            "PaymentMethod_Credit card (automatic)": 1 if request.form.get('payment') == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if request.form.get('payment') == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if request.form.get('payment') == "Mailed check" else 0
        }

        df = pd.DataFrame([data])

        # fill missing columns safely
        for col in columns:
            if col not in df.columns:
                df[col] = 0

        df = df[columns]

        # scale
        df_scaled = scaler.transform(df)

        # predict
        pred = model.predict(df_scaled)[0]

        result = f"{sim} Customer WILL CHURN ❌" if pred == 1 else f"{sim} Customer will NOT CHURN ✅"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)