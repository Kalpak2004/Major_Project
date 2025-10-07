import os
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load model from project root (ensure random_forest_model.pkl is in the same folder as app.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
model = joblib.load(MODEL_PATH)

def parse_form_values(form):
    keys = [
        'age',
        'fathersct',
        'mothersct',
        'family_history',
        'fatigue',
        'jaundice',
        'swelling_hands',
        'frequent_infection',
        'pain_crises',
        'hemoglobin',
        'hb_electrophoresis_percentage'
    ]
    values = []
    for k in keys:
        v = form.get(k, '0')
        try:
            values.append(float(v))
        except:
            values.append(0.0)
    X = np.array(values).reshape(1, -1)
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        X = parse_form_values(request.form)
        pred = model.predict(X)
        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                prob = float(model.predict_proba(X).max())
            except:
                prob = None

        if int(pred[0]) == 1:
            prediction_text = "Prediction: Likely sickle cell (based on inputs). Please consult a medical professional for confirmation."
        else:
            prediction_text = "Prediction: Unlikely sickle cell (based on inputs). If symptoms persist, consult a doctor."

        if prob is not None:
            prediction_text += f" (confidence: {prob:.2f})"

        return render_template('index.html', prediction=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
