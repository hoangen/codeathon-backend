import json
import os
import zipfile

from flask import Flask
from flask import request, send_from_directory

from .wide_deep.wide_deep import predict, predict_file

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['MODEL_FILE'] = 'model.zip'


@app.route('/predict', methods=['POST'])
def predict_income():
    if request.method == 'POST':
        predict_json = json.loads(request.data)
        predict_data = [item for item in predict_json]
        predict_result = predict(predict_data)

        return str(predict_result)

    return 'Bad request'


@app.route('/predict/file', methods=['POST'])
def predict_income_file():
    if 'file' not in request.files:
        return 'No File'

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    uploaded_file = request.files['file']
    if uploaded_file:
        filename = uploaded_file.filename
        file_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_full_path)

        app.logger.debug('File is saved as %s', file_full_path)

        predict_result = predict_file(os.path.abspath(file_full_path))

        return str(predict_result)

    return 'Bad request, no upload file'


@app.route('/model/upload', methods=['POST'])
def model_upload():
    print request.files
    if 'file' not in request.files:
        return 'No File'

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    uploaded_file = request.files['file']
    if uploaded_file:
        filename = uploaded_file.filename
        file_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_full_path)

        app.logger.debug('File is saved as %s', file_full_path)

        zip_ref = zipfile.ZipFile(file_full_path, 'r')
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])
        zip_ref.close()

        return 'Success'

    return 'No uploaded file'


@app.route('/model/download', methods=['GET'])
def model_download():

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               app.config['MODEL_FILE'],
                               mimetype='application/octet-stream')
