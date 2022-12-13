import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from recommender_system import get_recommendations_metadata



flask_app = Flask(__name__)


@flask_app.route("/")
def Home():
   
    return "hello"

@flask_app.route("/train-model", methods = ["POST"])
def train_model():
    exec(open("train_model.py").read())
    return "Model Trained"
    

@flask_app.route("/predict-auctions/<auction_id>", methods = ["GET"])
def predict(auction_id):
    predictedAuctions = get_recommendations_metadata(auction_id)
    
    response={"auctions":predictedAuctions}
  
    
    json_object = json.dumps(response)
    return json_object

if __name__ == "__main__":
    flask_app.run(debug=True)



