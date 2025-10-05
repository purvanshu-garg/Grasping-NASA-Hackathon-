from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures



app = Flask(__name__)



model = joblib.load("ensemble.pkl")

@app.route("/")   # Home route
def home():
    return render_template("indexgpt.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/csv_route") 
def csv():
    return render_template("CSV.html")

@app.route("/value_route") 
def value_enter():
    return render_template("goku.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not uploaded_file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        df = pd.read_csv(uploaded_file)
        
       
        # TODO: preprocess df properly before prediction
        prediction = model.predict(df.values)
        return render_template("results.html", predictions=prediction)
    except Exception as e:
        # Catch any error and return as JSON instead of crashing
        return jsonify({"error": str(e)}), 500
        
@app.route("/predict", methods=["POST"])
def predict_value():
    try:
        data = request.get_json()  # from fetch body

        # Extract values in correct order (must match training!)
        values = [
            float(data["orbitalPeriod"]),
            float(data["transitDuration"]),
            float(data["transitDepth"]),
            float(data["planetaryRadius"]),
            float(data["insolationFlux"]),
            float(data["equilibriumTemp"]),
            float(data["stellarTemp"]),
            float(data["stellarGravity"]),
            float(data["stellarRadius"])
        ]

        # Reshape into 2D array for scikit-learn
        arr = np.array(values).reshape(1, -1)

        prediction = model.predict(arr)

        return render_template("results.html", predictions=prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
