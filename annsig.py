# import all the libraries
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
import pickle

# import updated dataset
updatedBC = pd.read_csv('breast-cancer.csv')
updatedBC = updatedBC.drop("id", axis = 1)
updatedBC = updatedBC[(updatedBC['concavity_mean'] != 0)]

# taking X and y in the dataset
X_updatedBC = updatedBC.drop('diagnosis', axis = 1)
y_updatedBC = updatedBC['diagnosis']

# data normalization X
scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X_updatedBC)
X_normalized = pd.DataFrame(data = X_rescaled, columns = X_updatedBC.columns)
# Temp:
# X_normalized = X_updatedBC

# encoding y
set_of_classes = y_updatedBC.value_counts().index.tolist()
set_of_classes= pd.DataFrame({'diagnosis': set_of_classes})
y_encoded = pd.get_dummies(y_updatedBC)

# random state
random_num = random.randint(0, 100000)

# model
mlp = MLPClassifier(solver = 'sgd', random_state = random_num
, activation = 'logistic', learning_rate_init = 0.25, batch_size = 150, hidden_layer_sizes = (30, 10), max_iter = 500)


CV = cross_validate(mlp, X_normalized, y_encoded, cv=10, scoring=['accuracy', 'neg_mean_squared_error'])

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2)

mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)

# make pickle file of the model, mlp also used to fit data so have to pass mlp as the object
pickle.dump(mlp, open("annsig.pkl", "wb"))

print("Accuracy : ", accuracy_score(y_test, pred))
print("Mean Square Error : ", mean_squared_error(y_test, pred))

print(pred[:5])
print("Confusion Matrix for each label : ")
print(multilabel_confusion_matrix(y_test, pred))

print("Classification Report : ")
print(classification_report(y_test, pred))