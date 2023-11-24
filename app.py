import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

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
@app.route("/predict", methods = ["POST"])
def predict():
    # when we receive attributes of tumor from user, convert values into float
    float_features = [float(x) for x in request.form.values()]
    # store user inputs that are converted from float in array
    features = [np.array(float_features)]
    # normalize each value in features
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)
    # then call model with predict method and put in array features
    prediction = model.predict(rescaled_features)
    # user render template with index html and prediction text whether it is benign or malignant
    return render_template("index.html", prediction_text = "The tumor diagnosis is {}".format(prediction))



if __name__  == '__main__':
    app.run(debug = True)


# when run, will have it on local port, can access website from there