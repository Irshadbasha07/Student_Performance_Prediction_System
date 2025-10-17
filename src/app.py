from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__, template_folder="../templates", static_folder="../static")
MODEL_PATH = r"C:\Users\shaik abdulrasool\Downloads\final_year_project\final_year_project\saved_models\student_model.pkl"

def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"error":"Model not found. Please train the model by running src/model_training.py"}), 400
    try:
        parental = int(request.form.get("parental", 2))
        internet = 1 if request.form.get("internet","yes") == "yes" else 0
        study = float(request.form.get("study", 5.0))
        attend = float(request.form.get("attend", 80.0)) / 100.0
        prev = float(request.form.get("prev", 70.0)) / 100.0
        features = np.array([[parental, internet, study, attend, prev]])
        pred = model.predict(features)[0] * 100.0
        return render_template("result.html", prediction=round(pred,1), name=request.form.get("name","Student"), roll=request.form.get("roll","-"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)