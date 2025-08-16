from flask import Flask, request, jsonify
from fastai.vision.all import load_learner, PILImage
from flask_cors import CORS  # Only for local testing

app = Flask(__name__)

# Enable CORS for *any* origin while testing locally
CORS(app)

# Load model
learn = load_learner('export.pkl')

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/predict', methods=['POST'])
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)