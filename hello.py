import json
from flask import Flask
from flask import request
from wide_deep.wide_deep import predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_income():
    if request.method == 'POST':
        predict_json = json.loads(request.data)
        predict_data = [item for item in predict_json]
        predict_result = predict(predict_data)
        return str(predict_result)

    return 'Bad request'
