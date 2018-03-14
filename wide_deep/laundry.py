# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import itertools

import numpy as np
import tensorflow as tf
import os

_CSV_COLUMNS = [
    'Time',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount', 'Class'
]

_CSV_COLUMNS_PREDICT = [
    'Time',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount'
]

_CSV_COLUMN_DEFAULTS = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.]]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/tmp/laundry_data/laundry.train1',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/tmp/laundry_data/laundry.test1',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def root_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '../upload')


def build_model_columns():
    time = [tf.feature_column.numeric_column('Time')]
    v1_28 = [tf.feature_column.numeric_column("V" + str(i)) for i in range(1, 29)]
    amount = [tf.feature_column.numeric_column('Amount')]
    return time + v1_28 + amount


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
            feature_columns=columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=columns,
            dnn_feature_columns=columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have either run data_download.py or '
            'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing value', value)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('Class')
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    print(type(dataset))
    return dataset


def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(root_dir(), ignore_errors=True)
    model = build_estimator(root_dir(), FLAGS.model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


def predict(predict_data):
    print(type(predict_data))
    predict_length = len(predict_data)
    print("predict data length: " + str(predict_length))
    model = build_estimator(root_dir(), 'wide')
    zip_data = zip(*predict_data)
    zip_data = [np.asarray(item) for item in zip_data]
    features_predict = dict(zip(_CSV_COLUMNS_PREDICT, zip_data))
    print("type of feature " + str(type(features_predict)))
    input_fn_predict = tf.estimator.inputs.numpy_input_fn(
        x=features_predict,
        y=None,
        shuffle=True
    )
    y = model.predict(input_fn=input_fn_predict)
    probabilities = list(p['probabilities'] for p in itertools.islice(y, len(predict_data)))
    print("Predictions: {}".format(str(probabilities)))
    print(type(probabilities))
    return np.asarray(probabilities)


def predict_file(cvs_file):
    print("file path: " + cvs_file)
    model = build_estimator(root_dir(), 'wide')
    y = model.predict(input_fn=lambda: input_fn(cvs_file, 1, False, 1))
    return [np.argmax(p['probabilities'], 0) for p in y]


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
