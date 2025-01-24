from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Constants
IMG_SIZE = 64  # Image size should match your training size (64x64)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_parameters.npz")  # Path to saved parameters
CLASS_LABELS = {0: 'covid', 1: 'lung_opacity', 2: 'normal', 3: 'viral_pneumonia'}

# Load model parameters
def load_model_parameters(file_path):
    try:
        data = np.load(file_path)
        parameters = {key: tf.convert_to_tensor(value) for key, value in data.items()}
        return parameters
    except Exception as e:
        raise ValueError(f"Error loading model parameters: {e}")

# Forward propagation
def forward_propagation(X, parameters):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    #print(f"Z1: {Z1}")
    #print(f"A1: {A1}")
    #print(f"Z2: {Z2}")
    #print(f"A2: {A2}")
    #print(f"Z3: {Z3}")
    
    return Z3

# Preprocess input image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)).convert('L')  # Grayscale conversion
        img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
        img_flattened = img_array.flatten().reshape(-1, 1)  # Flatten to match model input
        return tf.convert_to_tensor(img_flattened, dtype=tf.float32)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

# Initialize Flask app
app = Flask(__name__)

# Load the model parameters globally
parameters = load_model_parameters(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions.
    Expects a POST request with an image file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save and preprocess the image
        temp_path = os.path.join("temp_image.jpg")  # Temporary path
        file.save(temp_path)
        input_data = preprocess_image(temp_path)
        os.remove(temp_path)  # Clean up the temporary file

        # Perform forward propagation
        logits = forward_propagation(input_data, parameters)

        # Debugging Step 1: Print the logits
        #print(f"Logits (Z3): {logits.numpy()}, Shape: {logits.shape}")

        # need to convert logits shape from (4,1) to (4,) before applying softmax
        logits = tf.reshape(logits, [-1])  # Flatten to 1D

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy().flatten()

        # Debugging Step 2: Print the softmax probabilities
        #print(f"Softmax probabilities: {probabilities}")
        
        predicted_class = np.argmax(probabilities)
        class_name = CLASS_LABELS[predicted_class]

        # Return the prediction as a response
        return jsonify({
            "predicted_class": class_name,
            "confidence": probabilities.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5002)
