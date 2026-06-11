from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # allow frontend on another port to call this API

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.txt")
LABEL_DIST_FILE = os.path.join(RESULTS_DIR, "label_distribution.csv")
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "predictions_sample.csv")
PLATFORM_SENT_FILE = os.path.join(RESULTS_DIR, "platform_sentiment.csv")


@app.route("/metrics")
def get_metrics():
    """Return accuracy, precision, recall, F1 as JSON."""
    metrics = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                try:
                    value = float(value.strip())
                except ValueError:
                    continue
                metrics[key] = value
    return jsonify(metrics)


@app.route("/label-distribution")
def get_label_distribution():
    """Return overall sentiment label distribution."""
    if not os.path.exists(LABEL_DIST_FILE):
        return jsonify([])
    df = pd.read_csv(LABEL_DIST_FILE)
    # Expected columns: sentiment_category, count
    return jsonify(df.to_dict(orient="records"))


@app.route("/platform-sentiment")
def get_platform_sentiment():
    """Return platform-wise sentiment distribution."""
    if not os.path.exists(PLATFORM_SENT_FILE):
        return jsonify([])
    df = pd.read_csv(PLATFORM_SENT_FILE)
    # Expected columns: platform, sentiment_category, count
    return jsonify(df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
