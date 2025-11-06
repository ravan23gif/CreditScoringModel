from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'credit_scoring_rf_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = np.array([[
            int(data['age']),
            int(data['job']),
            int(data['housing']),
            float(data['credit_amount']),
            int(data['duration']),
            int(data['savings']),
            int(data['checking']),
            int(data['credit_history']),
            int(data['purpose']),
            int(data['employment'])
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'risk': 'Good' if prediction == 1 else 'Bad',
            'probability': float(probability[1] * 100),
            'riskLevel': 'Low' if prediction == 1 else 'High',
            'recommendation': 'Approve' if prediction == 1 else 'Review Required',
            'confidence': float(max(probability) * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)