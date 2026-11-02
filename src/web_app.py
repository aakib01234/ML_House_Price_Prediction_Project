from flask import Flask, jsonify, render_template, request
from src.model_inference import predict

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    try:
        result = predict(payload)
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed. Check server logs."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
