import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create flask app
app = Flask(__name__)

# load pickle model
model = pickle.load(open("annsig.pkl", "rb"))

# home page route
@app.route("/")
# once on home page, calls function home which returns index html
def Home():
    return render_template("index.html")

# create route where predict method page is found
#use post as a way to recieve the values from the model
@app.route("/predict", method = ["POST"])
def predict():
    # when we recieve atttributes of tumor from user, convert values into float
    #
    float_features =