from flask import Flask, render_template, jsonify
import csv
from collections import Counter

app = Flask(__name__)

CSV_FILE = "posture_data.csv"

def read_data():
    timestamps = []
    neck_angles = []
    back_angles = []
    posture_labels = []

    try:
        with open(CSV_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestamps.append(row['Time'])
                neck_angles.append(float(row['Neck Angle']))
                back_angles.append(float(row['Back Angle']))
                posture_labels.append(row['Posture'])
    except:
        pass

    return timestamps, neck_angles, back_angles, posture_labels


@app.route("/")
def dashboard():
    timestamps, neck, back, posture = read_data()

    summary = Counter(posture)

    return render_template("index.html",
                           timestamps=timestamps,
                           neck=neck,
                           back=back,
                           summary=summary)


@app.route("/api/data")
def api_data():
    timestamps, neck, back, posture = read_data()

    return jsonify({
        "timestamps": timestamps,
        "neck": neck,
        "back": back,
        "posture": posture
    })


def run_flask():
    app.run(debug=True, use_reloader=False)