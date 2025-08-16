import sys
import pathlib
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

from flask import Flask, request, jsonify
from flask_cors import CORS
from fastai.vision.all import load_learner, PILImage
from io import BytesIO
import base64
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app, origins=["https://evanblankenheim.com"])

# --- RATE LIMITING CONFIG ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]  # e.g., 10 requests/minute per IP
)

# --- SECURITY: ORIGIN CHECK ---
API_SECRET = "<SECRET_KEY_GENERATOR>"
ALLOWED_ORIGINS = ["https://evanblankenheim.com", "http://localhost:3000"]

@app.before_request
def restrict_origin():
    # Allow preflight OPTIONS requests without blocking
    if request.method == "OPTIONS":
        return

    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return jsonify({"error": "Forbidden - Invalid origin"}), 403

    # Check for custom API secret header
    if request.headers.get("X-API-SECRET") != API_SECRET:
        return jsonify({"error": "Forbidden - Invalid API secret"}), 403

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # Include your custom header!
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-SECRET"
    return response


learn = load_learner('export.pkl')

# --- ROUTES ---
@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")  # Optional: Stricter limit for this route
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400

    f = request.files['file']
    img_bytes = f.read()

    try:
        img = PILImage.create(img_bytes)
        pred, pred_idx, probs = learn.predict(img)
        return jsonify({
            'prediction': str(pred),
            'pred_index': int(pred_idx),
            'probability': float(probs[pred_idx])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)