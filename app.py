from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load trained model
model = joblib.load("model/xgboost_model.pkl")

# Load feature names
feature_names = joblib.load("model/feature_names.pkl")


@app.route("/")
def home():
    return render_template("index.html",feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():

    input_data = []

    for feature in feature_names:
        value = float(request.form[feature])
        input_data.append(value)

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    prediction = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Solar Power : {prediction} kW",
        feature_names=feature_names

    )


if __name__ == "__main__":
    app.run(debug=True)