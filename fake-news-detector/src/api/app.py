import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../..'))

from flask import Flask, request, jsonify
import joblib
from src.data.loader import load_config
from src.features.preprocess import preprocess_texts

app = Flask(__name__)

# Load model and config with error handling
try:
    model = joblib.load("models/tfidf_logreg.joblib")
    cfg = load_config()
except FileNotFoundError as e:
    print(f"Error loading model or config: {e}")
    model = None
    cfg = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "tfidf_logreg.joblib"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector API is running!", "endpoints": ["/health", "/predict"]})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or cfg is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        payload = request.get_json(force=True)
        title = payload.get("title", "")
        body = payload.get("body", "")
        
        if not title and not body:
            return jsonify({"error": "Title or body is required"}), 400
        
        text = (title + " . " + body).strip()
        proc = preprocess_texts([text], cfg)

        # Predictions
        clf = model.named_steps["clf"]
        proba = None
        try:
            proba = clf.predict_proba(model.named_steps["vect"].transform(proc))[0]
        except Exception as e:
            print(f"Probability calculation error: {e}")
            proba = None

        pred = model.predict(proc)[0]
        result = {
            "label": pred, 
            "confidence": float(proba.max()) if proba is not None else None
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)