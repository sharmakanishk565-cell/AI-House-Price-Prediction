
from flask import Flask, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("house_price_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "AI House Price Prediction Running Successfully"

@app.route('/predict')
def predict():

    sample_data = [[7420,4,2,3,1,0,0,0,1,2,1,2]]

    scaled_data = scaler.transform(sample_data)

    prediction = model.predict(scaled_data)

    return jsonify({
        "Predicted House Price": round(prediction[0], 2)
    })

if __name__ == '__main__':
    app.run(debug=True)

