# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:15:16 2021

@author: lukas
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Daten einlesen und vorbereiten
dataset = pd.read_csv(r"./covid_early_stage_symptoms.csv")
dataset["gender"].replace("female", 0, inplace = True)
dataset["gender"].replace("male", 1, inplace = True)
labels = dataset["SARS-CoV-2 Positive"]
features = pd.DataFrame(dataset)
del features["SARS-CoV-2 Positive"]

#Traininng und Testdaten erstellen
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)

#Modell Trainieren
rf = RandomForestClassifier(criterion="entropy",n_estimators=100,random_state=100)
rf.fit(train_features,train_labels)

predictions = rf.predict(test_features)

#Auswertung des Modells 
print(confusion_matrix(test_labels,predictions))
print(classification_report(test_labels,predictions))
print("Accuracy:"+str(accuracy_score(test_labels,predictions )))


import joblib

joblib.dump(rf, "./self_pred.joblib")