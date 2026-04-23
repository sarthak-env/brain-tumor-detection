from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_vgg16.h5")  # ← updated from brain_tumor_cnn.h5
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224                                              # ← updated from 128
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}

# ── Load model once at startup ────────────────────────────────────────────────
log.info("Loading model from %s …", MODEL_PATH)
model = load_model(MODEL_PATH)
log.info("Model loaded successfully.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      "brain_tumor_vgg16",
        "classes":    CLASSES,
        "image_size": IMAGE_SIZE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send as multipart/form-data with key 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "unknown"
        return jsonify({"error": f"File type '.{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}"}), 415

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        return jsonify({"error": f"File too large. Max {MAX_FILE_BYTES // (1024*1024)} MB."}), 413

    try:
        arr         = preprocess_image(image_bytes)
        predictions = model.predict(arr, verbose=0)[0]

        scores = {
            CLASSES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASSES))
        }
        top_class  = max(scores, key=scores.get)
        confidence = scores[top_class]
        label      = "No Tumor Detected" if top_class == "notumor" \
                     else f"Tumor Detected: {top_class.capitalize()}"

        log.info("Prediction: %s (%.2f%%)", top_class, confidence)
        return jsonify({
            "prediction": top_class,
            "confidence": confidence,
            "label":      label,
            "scores":     scores,
        })

    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


if __name__ == "__main__":
    print("\n🧠  NeuroScan AI — Backend")
    print("─" * 36)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Classes: {CLASSES}")
    print(f"  ImgSize: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("─" * 36)
    print("  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)