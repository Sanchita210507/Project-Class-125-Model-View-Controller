from flask import Flask, jsonify, request
from classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def intro():
    return "Hey! This is Sanchita!!"

@app.route("/predict-alphabet", methods = ["POST"])
def predict_data():
    image = request.files.get("alphabet")
    pred = get_prediction(image)
    return jsonify({
        "prediction": pred
    }), 200

if __name__ == "__main__":
    app.run(debug = True, port = 8000)