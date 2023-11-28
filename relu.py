import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
import pickle

dataset = pd.read_csv("breast-cancer.csv")
dataset = dataset.drop("id", axis = 1)
dataset = dataset[(dataset['concavity_mean'] != 0 )]

X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']

scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X = pd.DataFrame(data = X_rescaled, columns = X.columns)

set_of_classes = y.value_counts().index.tolist()
set_of_classes= pd.DataFrame({'diagnosis': set_of_classes})
y = pd.get_dummies(y)

mlp = MLPClassifier(solver = 'sgd', random_state = 0
, activation = 'relu', learning_rate_init = 0.2, batch_size = 225, hidden_layer_sizes = (30, 3), max_iter = 500)

CV = cross_validate(mlp, X, y, cv=10, scoring=['accuracy', 'neg_mean_squared_error'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)

# make pickle file of the model, mlp also used to fit data so have to pass mlp as the object
pickle.dump(mlp, open("relu.pkl", "wb"))

print("Accuracy : ", accuracy_score(y_test, pred))
print("Mean Square Error : ", mean_squared_error(y_test, pred))

print(pred[:5])
print("Confusion Matrix for each label : ")
print(multilabel_confusion_matrix(y_test, pred))

print("Classification Report : ")
print(classification_report(y_test, pred))
