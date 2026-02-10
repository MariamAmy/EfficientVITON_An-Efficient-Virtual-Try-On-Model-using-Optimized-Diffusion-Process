from flask import Flask, request, jsonify
from inference import Model
import os
import base64
import subprocess
from PIL import Image
from io import BytesIO
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

initialized = False

model = Model()

BASE_DIR = "/home/user/stableVTON"  # Set your actual path
DATA_DIR = os.path.join(BASE_DIR, "StableVITON/DATA/zalando-hd-resized/inference")
PREPROCESSING_DIR = os.path.join(BASE_DIR, "preprocessing")

def base64_to_image(base64_string):
    """Converts a base64 string to a PIL image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def execute_shell_command(command):
    """Executes a shell command and returns the output or raises an exception."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Command failed: {command}\nError: {stderr.decode()}")
    return stdout.decode().strip()

@app.route('/init', methods=['POST'])
def init_endpoint():
    global initialized
    if initialized:
        return jsonify({'message': 'Model already initialized'}), 200
    

    model.initialize_model()
    initialized = True
    return jsonify({'message': 'Model initialized'}), 200

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    data = request.get_json()
    if not data or 'cloth_name' not in data:
        return jsonify({'error': 'Please provide a cloth_name in the request body'}), 400

    cloth_name = data['cloth_name']

    with open("./DATA/zalando-hd-resized/inference_pairs.txt", "w") as f:
        f.write(f"test.jpg {cloth_name}.jpg")
    
    try:
        encoded_image = model.run_inference()
        return jsonify({'image': encoded_image}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        filename = data.get('filename', 'image')

        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        # 1. Save the Image to the data folder
        image = base64_to_image(image_base64)
        image = image.convert("RGB")
        image = image.resize((768, 1024))
        image_path = os.path.join(DATA_DIR, "image", f"{filename}.jpg")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
        print(f"Image saved to {image_path}")
        
        # 2. Execute the sh file
        sh_file_path = os.path.join(PREPROCESSING_DIR, "preprocess.sh")
        start_time = time.time()
        execute_shell_command(sh_file_path)
        end_time = time.time()

        # Calculate and display total time taken
        total_time = end_time - start_time
        print(f"Total time taken for all processes: {total_time} seconds")

        # 3. Return 200 OK
        return jsonify({'message': 'Preprocessing finished successfully'}), 200


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)