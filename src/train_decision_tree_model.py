import numpy as np
from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn import metrics
import pickle


# Training the model
# Getting data and splitting into training and testing sets
data_dict = {}
with open('training_data.json', 'r') as f:
    data_dict = json.load(f)

X = np.zeros((len(data_dict), len(data_dict["101"]["features"])))
y = []

i = 0
for patient_id in data_dict:
    feature_list = []
    lung_condition = data_dict[patient_id]["lung_condition"]
    for feature in data_dict[patient_id]["features"]:
        feature_list.append(data_dict[patient_id]["features"][feature])
    X[i] = np.array(feature_list)
    y.append(lung_condition)
    i += 1

# Class representation in X, y is highly imbalanced
# transform the dataset
counter = Counter(y)
print(counter)
oversample = BorderlineSMOTE()
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model using an ensemble of decision trees
# The number of trees can be set via the “n_estimators” argument and defaults to 100
dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred1 = dtree_model.predict(X_test)

# Training the model using gradient boosted decision trees
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
y_pred2 = classifier.predict(X_test)


# Calculating recall and precision using scikit_learn:
# Print the confusion matrix
print("Using bagged decision trees...")
print(metrics.confusion_matrix(y_test, y_pred1))
    
# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_pred1, digits=3))
    
print("Using Gradient Boosted Decision trees...")
print(metrics.confusion_matrix(y_test, y_pred2))
print(metrics.classification_report(y_test, y_pred2, digits=3))

# save the gradient boosting classifier
filename = 'gradient_boosting_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))
# save the bagged decision tree classifier
filename = 'bagged_decision_tree.sav'
pickle.dump(dtree_model, open(filename, 'wb'))
