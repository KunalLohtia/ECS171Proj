import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

# create flask app
app = Flask(__name__)

# load pickle model
model = pickle.load(open("relu.pkl", "rb"))

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
    # print(row)
    features = np.array([float(x) for x in row.split(',')])
    # store user inputs that are converted from float in array
    # print(features)
    # normalize each value in features
    rescaled_features = scaler.transform(features.reshape(1, -1))
    # print(rescaled_features)
    # then call model with predict method and put in array features
    prediction = model.predict(rescaled_features)
    # print(prediction)
    # user render template with index html and prediction text whether it is benign or malignant
    return render_template("index.html", prediction_text = "The tumor diagnosis is {}".format(prediction))



if __name__  == '__main__':
    app.run(debug = True)


# when run, will have it on local port, can access website from there