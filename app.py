import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

# create flask app
app = Flask(__name__)

# load pickle models
relu = pickle.load(open("relu.pkl", "rb"))
sigmoid = pickle.load(open("annsig.pkl", "rb"))
logistic = pickle.load(open("logistic.pkl", "rb"))


# import and update dataset as done in training
updatedBC = pd.read_csv('breast-cancer.csv')
updatedBC = updatedBC.drop("id", axis = 1)
updatedBC = updatedBC[(updatedBC['concavity_mean'] != 0)]
X_updatedBC = updatedBC.drop('diagnosis', axis = 1)

# initialize data scaler/normalizer for X
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_updatedBC)

# home page route
@app.route("/")
# once on home page, calls function home which returns index html
def Home():
    return render_template("index.html")

# create route where predict method page is found
#use post as a way to recieve the values from the model
@app.route("/predict", methods = ["POST"])
def predict():
    # when we receive attributes of tumor from user, convert values into float
    row = request.form.get('List of Attributes')
    features = np.array([float(x) for x in row.split(',')])
    # store user inputs that are converted from float in array
    # normalize each value in features
    rescaled_features = scaler.transform(features.reshape(1, -1))
    # then call each model with predict method and put in rescaled features
    prediction_relu = relu.predict(rescaled_features)
    prediction_relu = "benign" if prediction_relu[0][0] == 1 else "malignant"
    prediction_sigmoid = sigmoid.predict(rescaled_features)
    prediction_sigmoid = "benign" if prediction_sigmoid[0][0] == 1 else "malignant"
    prediction_logistic = logistic.predict(rescaled_features)
    prediction_logistic = "benign" if prediction_logistic == "B" else "malignant"
    # user render template with index html and prediction text whether it is benign or malignant
    return render_template("index.html",
                prediction_text_relu = "The ReLU ANN's tumor diagnosis is {}".format(prediction_relu),
                prediction_text_sigmoid = "The sigmoid ANN's tumor diagnosis is {}".format(prediction_sigmoid),
                prediction_text_logistic = "The logistic regression model's tumor diagnosis is {}".format(prediction_logistic))



if __name__  == '__main__':
    app.run(debug = True)


# when run, will have it on local port, can access website from there