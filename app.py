"""
MLBSharp Web Dashboard
Serves the daily strikeout report as a mobile-friendly web app.
"""

import json
import os
from datetime import date
from flask import Flask, render_template, jsonify

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA = os.path.join(SCRIPT_DIR, "dashboard_data.json")


def load_dashboard_data() -> dict:
    """Load the latest dashboard data."""
    try:
        with open(DASHBOARD_DATA) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "date": date.today().strftime("%B %d, %Y"),
            "generated_at": "",
            "all_pitchers": [],
            "total_pitchers": 0,
        }


@app.route("/")
def index():
    data = load_dashboard_data()
    return render_template("index.html", data=data)


@app.route("/api/data")
def api_data():
    return jsonify(load_dashboard_data())


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
