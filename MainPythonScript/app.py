import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

# Load trained model
model = joblib.load("rainfall.pkl")


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=["POST"])
def predict():

    # Get form data
    data = request.form.to_dict()

    # Convert empty strings to NaN
    for key in data:
        if data[key] == "":
            data[key] = np.nan

    # Numeric columns (must match training dataset EXACTLY)
    numeric_cols = [
        'MinTemp','MaxTemp','Rainfall','WindGustSpeed',
        'WindSpeed9am','WindSpeed3pm',
        'Humidity9am','Humidity3pm',
        'Pressure9am','Pressure3pm',
        'Temp9am','Temp3pm',
        'Year','Month','Day'
    ]

    # Convert numeric columns to float
    for col in numeric_cols:
        if col in data and data[col] is not np.nan:
            try:
                data[col] = float(data[col])
            except:
                data[col] = np.nan

    # Convert dictionary to DataFrame
    df = pd.DataFrame([data])
    df['@'] = 0

    # Make prediction
    prediction = model.predict(df)[0]

    # Optional: Probability
    try:
        probability = model.predict_proba(df)[0][1] * 100
    except:
        probability = None

    # Render result page
    if prediction in ["Yes", 1]:
        return render_template("chance.html", prob=probability)
    else:
        return render_template("nochance.html", prob=probability)


if __name__ == "__main__":
    app.run(debug=True)
