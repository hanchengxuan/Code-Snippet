from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from datetime import datetime
import csv
import json
import numpy as np
import pandas as pd
from waitress import serve

from model.utils import output_predictions
from model.train_predict import train_policy_value_network, predict_policy_value_network

import conf.global_settings as settings

import logging.config

app = Flask(__name__)

logging.config.fileConfig('conf/log.conf')
log = logging.getLogger('simple')

authenticated_status = False
testFilePath = None
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_valid_csv(file_path):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            return True
    except Exception as e:
        log.error(f'Invalid CSV: {e}')
        return False
    
def validate_pin(user_pin):
    with open('conf/pin.txt', 'r', encoding='utf-8') as file:
        pins = file.read().splitlines()
        if user_pin in pins:
            return True
        else:
            return False
        
def save_output(username, action, message, result=None):
    output_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "output.txt"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'a', encoding='utf-8') as file:
        data = {
            'username': username,
            'action': action,
            'message': message,
            'timestamp': timestamp
        }
        if result:
            data['result'] = result
        file.write(json.dumps(data) + '\n')

    return file_path


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/index.html')
def index():
    return render_template('index.html', userName=settings.current_user, settings=settings)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    return jsonify(settings.available_models)

@app.route('/set_current_model', methods=['POST'])
def set_current_model():
    data = request.get_json()
    model = data.get('model')

    if model in settings.available_models:
        settings.current_model = model
        settings.update()
        log.debug(f'Current model set to {model}')
        log.debug(f'Current model set to settings {settings.current_model}')
        return jsonify({'message': f'Current model set to {settings.available_models[model]}'})
    else:
        return jsonify({'error': 'Invalid model selection'}), 400
    
@app.route('/get_default_predict_data', methods=['GET'])
def get_default_predict_data():
    predict_file_path = settings.prediction_file
    try:
        with open(predict_file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return jsonify({'error': 'Prediction file not found'}), 404
    
@app.route('/get_last_predict_result', methods=['GET'])
def get_last_predict_result():
    output_file_path = settings.output_file
    try:
        with open(output_file_path, 'r', encoding='utf-8') as file:
            results = file.readlines()
            results = [line.strip() for line in results]
            return jsonify(results)
    except FileNotFoundError:
        return jsonify([])

@app.route('/upload_predict_data', methods=['POST'])
def upload_predict_data():
    if 'Authorization' not in request.headers:
        return jsonify({'error': 'Please authenticate login'})
    
    global userName
    global testFilePath

    # Create a folder named after userName
    user_folder_path = os.path.join(os.getcwd(), f"{settings.user_data_dir}/{userName}")
    file_folder_path = os.path.join(user_folder_path, 'test')
    if not os.path.exists(file_folder_path):
        os.makedirs(file_folder_path)

    data = request.form['predictData']
    
    # Write data directly to CSV file
    testFilePath = os.path.join(file_folder_path, 'predict_data.csv')
    with open(testFilePath, 'w', newline='', encoding='utf-8') as file:
        file.write(data)

    log.info(f'Predict data saved to {testFilePath}')
    return jsonify(message='Data saved to ' + testFilePath)

@app.route('/predict', methods=['POST'])
def predict_data():
    if 'Authorization' not in request.headers:
        return jsonify({'error': 'Please authenticate login first'}), 401

    try:
        global testFilePath
        if testFilePath:
            # If the user uploaded a prediction data file, use the user-uploaded file for prediction
            log.debug(f'Predicting model with test file1: {testFilePath}')
            policies, values = predict_policy_value_network(predict_input_file=testFilePath)
        else:
            # If the user did not upload a prediction data file, use the default prediction file
            log.debug(f'Predicting model with default file')
            policies, values = predict_policy_value_network()

        # Assuming the code to call the prediction model has been executed elsewhere and the results are saved to output.txt
        output_file_path = settings.output_file

        # Read the OUTPUT file
        log.debug(f'Predicting model with output file: {output_file_path}')
        with open(output_file_path, 'r', encoding='utf-8') as file:
            
            # Read all lines
            results = file.readlines()  

            # Remove whitespace characters from each line
            results = [line.strip() for line in results]  
            
            log.debug(f'Model predict result: {results}')

        log.info(f'Model predict result: {results}')
        return jsonify(results)

    except FileNotFoundError:
        log.error(f'Output file not found: {output_file_path}')
        return jsonify({'error': 'Prediction result file not found'}), 404
    except Exception as e:
        log.error(f'Failed to predict model: {e}')
        return jsonify({'error': f'Failed to predict data: {e}'}), 500

@app.route('/auth', methods=['POST'])
def authenticate():
    global authenticated_status
    global userName
    data = request.json
    user_pin = data.get('pin')

    if validate_pin(user_pin):
        authenticated_status = True
        message = 'Authentication successful'
        userName = user_pin

        # Update settings.current_user
        settings.current_user = user_pin  

        log.debug(f'User {userName} authenticated')
        log.info(f'Action: authenticated, Message: {message}, user_pin: {userName}, Timestamp: {timestamp}')
        settings.update()

        return jsonify({'message': 'Authentication successful', 'user_pin': user_pin})
    else:
        message = f'Invalid pin: {user_pin}'
        log.info(f'Action: authenticated, Message: {message}, Timestamp: {timestamp}')
        return jsonify({'error': 'Incorrect password'})

# Clear userName in the logout() function
@app.route('/logout', methods=['POST'])
def logout():
    global authenticated_status
    global userName

    # Clear settings.current_user
    settings.current_user = '' 

    authenticated_status = False
    userName = ''

    settings.update()

    return jsonify({'message': 'Logout successful'})

if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=settings.tair_default_port, threaded=True)
    serve(app, host='0.0.0.0', port=settings.tair_default_port, threads=True)