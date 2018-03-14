import json
import os
from flask import Flask
from flask import request
from wide_deep.wide_deep import predict

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = '/home/hoangen/upload'

@app.route('/predict', methods=['POST'])
def predict_income():
    if request.method == 'POST':
        predict_json = json.loads(request.data)
        predict_data = [item for item in predict_json]
        predict_result = predict(predict_data)

        return str(predict_result)

    return 'Bad request'


@app.route('/model/upload', methods=['POST'])
def model_upload():
    print request.files
    if 'file' not in request.files:
        return 'No File'

    uploaded_file = request.files['file']
    if uploaded_file:
        filename = uploaded_file.filename
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        app.logger.debug('File is saved as %s', filename)
        print filename
        return 'Success'

    return 'No uploaded file'
