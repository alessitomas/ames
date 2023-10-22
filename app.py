from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)


with open('ridge_model.pkl', 'rb') as file:
    model = pickle.load(file)


def preprocess_input(input_features):
    return np.array([input_features])


@app.route('/ames/predict', methods=['POST'])
def predict_house_value():
    input_features = request.json['features']
    preprocessed_input = preprocess_input(input_features)
    prediction = model.predict(preprocessed_input)
    return jsonify({'predicted_value': prediction[0]})
    
if __name__ == '__main__':
    app.run(debug=True)
