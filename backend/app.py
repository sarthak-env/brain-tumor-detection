from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_cnn.h5')

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Must match the order from your training (train_generator.class_indices)
# {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = 128  # Must match what the model was trained on


# ── Helper: preprocess image ────────────────────────────────────────────────
def preprocess_image(image_bytes):
    """
    Takes raw image bytes, returns a numpy array ready for model prediction.
    Same preprocessing as training: resize to 128x128, normalize to 0-1.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (handles grayscale MRI images and PNGs with alpha)
    img = img.convert('RGB')

    # Resize to match training size
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert to numpy array and normalize (0-255 → 0-1)
    arr = np.array(img) / 255.0

    # Add batch dimension: (128, 128, 3) → (1, 128, 128, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Check if the server and model are running."""
    return jsonify({
        'status': 'ok',
        'model': 'brain_tumor_cnn',
        'classes': CLASSES,
        'image_size': IMAGE_SIZE
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a brain MRI image and returns classification results.

    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image (file)

    Response:
        {
            "prediction": "glioma",
            "confidence": 87.3,
            "scores": {
                "glioma": 87.3,
                "meningioma": 5.2,
                "notumor": 4.8,
                "pituitary": 2.7
            }
        }
    """
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded. Send image as multipart/form-data with key "image"'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Check file type
    allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return jsonify({'error': f'File type .{ext} not supported. Use: {allowed_extensions}'}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess
        input_array = preprocess_image(image_bytes)

        # Run prediction
        predictions = model.predict(input_array, verbose=0)[0]

        # Convert to percentages
        scores = {
            CLASSES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        # Get top prediction
        top_class = max(scores, key=scores.get)
        confidence = scores[top_class]

        return jsonify({
            'prediction': top_class,
            'confidence': confidence,
            'scores': scores
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🧠 NeuroScan AI Backend")
    print("─────────────────────────────")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASSES}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("─────────────────────────────")
    print("Server running at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
