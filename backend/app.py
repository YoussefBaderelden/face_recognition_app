from flask import Flask, request, jsonify, send_file
import os
import subprocess
import time
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
FACES_DIR = os.path.join(BACKEND_DIR, 'Faces')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(BACKEND_DIR, filename)
        file.save(temp_path)

        # Run MATLAB script with detailed output
        result = subprocess.run([
            "matlab",
            "-batch",
            f"addpath('{BACKEND_DIR}'); classify_image_simple('{temp_path}'); exit;"
        ], capture_output=True, text=True, check=True)

        print(f"MATLAB stdout: {result.stdout}")
        print(f"MATLAB stderr: {result.stderr}")

        os.remove(temp_path)

        # Wait a bit to ensure file is saved by MATLAB
        time.sleep(1)

        result_filename = os.path.splitext(filename)[0] + '_output' + os.path.splitext(filename)[1]
        result_path = os.path.join(RESULTS_DIR, result_filename)

        # Check if result file exists
        if not os.path.exists(result_path):
            print(f"Result file not found at: {result_path}")
            print(f"Contents of results directory: {os.listdir(RESULTS_DIR)}")
            return jsonify({'error': 'Processing completed but result file not found', 'path': result_path}), 500

        return send_file(
            result_path,
            mimetype=f'image/{os.path.splitext(filename)[1][1:]}',
            as_attachment=False
        )

    except subprocess.CalledProcessError as e:
        return jsonify({
            'error': 'MATLAB processing failed',
            'details': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }), 500
    except Exception as e:
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    if 'name' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Name and image required'}), 400

    name = request.form['name']
    file = request.files['image']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        existing_images = len([
            f for f in os.listdir(person_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        filename = f"{name}_{existing_images + 1}.jpg"
        filepath = os.path.join(person_dir, filename)
        file.save(filepath)

        # Run training with detailed error capture
        result = subprocess.run([
            "matlab",
            "-batch",
            f"addpath('{BACKEND_DIR}'); train_classifier(); exit;"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({
                'error': 'MATLAB processing failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }), 500

        return jsonify({'status': 'Training completed', 'person': name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_results', methods=['GET'])
def check_results():
    """Endpoint to check contents of results directory for debugging"""
    try:
        files = os.listdir(RESULTS_DIR)
        return jsonify({
            'results_dir': RESULTS_DIR,
            'files': files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({
        'status': 'Server is running',
        'results_dir': RESULTS_DIR,
        'faces_dir': FACES_DIR
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)