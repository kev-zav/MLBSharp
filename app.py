"""
MLBSharp Web Dashboard
Serves the daily strikeout report as a mobile-friendly web app.
"""

import csv
import json
import os
from datetime import date
from flask import Flask, render_template, jsonify

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA = os.path.join(SCRIPT_DIR, "dashboard_data.json")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
MODEL_START_DATE = "2026-04-11"


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


@app.route("/api/results")
def api_results():
    rows = []
    try:
        with open(RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date", "") < MODEL_START_DATE:
                    continue
                projected = row.get("projected_ks", "").strip()
                actual = row.get("actual_ks", "").strip()
                ip = row.get("innings_pitched", "").strip()
                rows.append({
                    "date": row.get("date", ""),
                    "pitcher_name": row.get("pitcher_name", ""),
                    "pitcher_hand": row.get("pitcher_hand", ""),
                    "team": row.get("team", ""),
                    "opponent": row.get("opponent", ""),
                    "projected_ks": float(projected) if projected else None,
                    "actual_ks": int(float(actual)) if actual else None,
                    "innings_pitched": float(ip) if ip else None,
                })
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return jsonify(rows)


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
