# filename: app.py

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our refactored inference logic
import inference as model_logic

# Initialize the Flask application
app = Flask(__name__)
CORS(app) 

# --- MODEL LOADING ---
print("Loading model...")
try:
    model = model_logic.load_model()
    print("Model loaded successfully. Ready for predictions.")
except Exception as e:
    print(f"FATAL: Application failed to start. Error loading model: {e}")
    model = None

# --- API ROUTES ---
@app.route('/', methods=['GET'])
def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return jsonify({
        "status": "success",
        "message": "Drone Detection API is running."
    })

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
            
    try:
        image_bytes = file.read()
        predictions = model_logic.get_drone_predictions(image_bytes, model)
        
        return jsonify({
            "status": "success",
            "predictions": predictions
        })
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"status": "error", "message": "Failed to process the image."}), 500