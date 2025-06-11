from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('ai/model.pkl')
except:
    model = None

@app.route("/")
def home():
    return "Aguero Super Tips API is Running 🚀"

@app.route("/predict", methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        proba = model.predict_proba(features)

        return jsonify({
            'prediction': prediction.tolist(),
            'confidence': proba.max().item()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
