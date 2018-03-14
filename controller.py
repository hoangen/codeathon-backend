import json
import csv
import os
import shutil
import zipfile

from flask_cors import CORS
from flask import Flask, Response
from flask import request, send_from_directory

from .wide_deep.laundry import predict, predict_file
from .wide_deep.wide_deep import predict as predict_legacy
from .wide_deep.wide_deep import predict_file as predict_file_legacy

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['MODEL_FILE'] = 'model.zip'


def root_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


def remove_first_line(filename):
    with open(filename, 'r') as fin:
        data = fin.read().splitlines(True)

    with open(filename + '1', 'w') as fout:
        fout.writelines(data[1:])


@app.route('/predict_legacy', methods=['POST'])
def predict_income():
    if request.method == 'POST':
        predict_json = json.loads(request.data)
        predict_data = [item for item in predict_json]
        predict_result = predict_legacy(predict_data)

        return str(predict_result)

    return 'Bad request'


@app.route('/predict/file_legacy', methods=['POST'])
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
        remove_first_line(file_full_path)

        predict_result = predict_file_legacy(os.path.abspath(file_full_path + '1'))

        with open(file_full_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row, result in zip(rows, predict_result):
            row['fraud'] = result

        return Response(json.dumps(rows), content_type="application/json")

    return 'Bad request, no upload file'


@app.route('/predict', methods=['POST'])
def predict_laundry():
    if request.method == 'POST':
        predict_json = json.loads(request.data)
        predict_data = [item for item in predict_json]
        predict_result = predict(predict_data)

        return str(predict_result)

    return 'Bad request'


@app.route('/predict/file', methods=['POST'])
def predict_laundry_file():
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
        remove_first_line(file_full_path)

        predict_result = predict_file(os.path.abspath(file_full_path + '1'))
        print(type(predict_result))
        with open(file_full_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row, result in zip(rows, predict_result):
            row['fraud'] = result.item()

        return Response(json.dumps(rows), content_type="application/json")

    return 'Bad request, no upload file'


@app.route('/model/upload', methods=['POST'])
def model_upload():
    print request.files
    if 'file' not in request.files:
        return 'No File'

    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)

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


@app.route('/model/download/<filename>', methods=['GET'])
def model_download_file(filename):
    app.logger.debug('Filename is %s', filename)

    content = get_file(filename)
    return Response(content, mimetype="application/zip")
