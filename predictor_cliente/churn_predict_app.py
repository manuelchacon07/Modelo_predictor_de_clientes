import pickle
import pandas as pd

from flask import Flask, jsonify, request

from churn_predict_service import predict_single

app = Flask('predict')

with open('models/potencial-client-model.pck', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()
    customer = pd.DataFrame.from_dict(customer, orient="index")
    customer = customer.transpose()
    prediction = predict_single(customer, model)
    result = {
    'Converted': bool(prediction),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
