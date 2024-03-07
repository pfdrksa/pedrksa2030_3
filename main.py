from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open('heart.pkl', 'rb'))  # Make sure this is the correct model file

app = Flask(__name__)


@app.route('/')
def index():
    return "Heart Disease Prediction API"


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve feature values from the request
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    # Convert to numpy array and reshape to match the input format
    input_query = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                           dtype=float)

    # Predict and return result
    result = model.predict(input_query)[0]
    return jsonify({'heart_disease_prediction': int(result)})


if __name__ == '__main__':
    app.run(debug=True)
