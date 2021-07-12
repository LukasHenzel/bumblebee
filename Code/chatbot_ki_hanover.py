# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:18:50 2021

@author: lukas
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle

#HanoverTagger initalisieren
from HanTa import HanoverTagger as ht
hanover = ht.HanoverTagger('morphmodel_ger.pgz')

#Daten einlesen
intents_file = open('./intents.json', encoding='utf-8').read()
intents = json.loads(intents_file, strict=False)

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']


#Komplette JSON Datei verarbeiten
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Tokenizing der Wörter welche in den Pattern vorkommen
        word = nltk.word_tokenize(pattern)
        words.extend(word)        
        #Zuordnung zwischen Tags und Wörtern
        documents.append((word, intent['tag']))
        # Erzeugung einer Liste mit allen Klassen
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
# Lemmatisierung mit HanTa
words_lemma=[]
for word in words:
    lemma = [lemma for (word,lemma,pos) in hanover.tag_sent(word.split())]
    words_lemma.append(' '.join(lemma))
words_lemma = sorted(list(set(words_lemma)))
classes = sorted(list(set(classes)))
# Durch documents werden die Patterns mit den Tags verbunden 
print (len(documents), "mit Tags verbundene Patterns")
# Als classes werden die Tags gespeichert
print (len(classes), "Klassenanzahl und Klassen:", classes)
# words_lemma beschreibt die Gruppe der einzigartigen Wörter
print (len(words_lemma), "Vokabularkänge mit folgenden Wörtern:", words_lemma)
pickle.dump(words_lemma,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# Trainignsdaten erstellen
training = []
# Leeres Output Array
output_empty = [0] * len(classes)
# BagofWords (BoW)
for doc in documents:
    bag = []
    # Tokenizing der Daten
    word_patterns = doc[0]
    # Lemmatisierung der Wörter, um diese vergleichbar zu machen
    word_patterns_lemma = []
    for word in word_patterns:
        lemma = [lemma for (word,lemma,pos) in hanover.tag_sent(word.split())]
        word_patterns_lemma.append(' '.join(lemma)) 
    # Wenn ein Wort in Patterns vorhanden ist wird eine 1 hinzugefügt, d. h. für jedes vorhandene Wort wird dem Array eine 1 hinzugefügt
    for word in words_lemma:
        bag.append(1) if word in word_patterns_lemma else bag.append(0)   
    # Der Output ist 0 für alle Tags welche ungleich zu dem aktuellen Tag sind
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
# Durchmischen der Trainigsdaten
random.shuffle(training)
training = np.array(training, dtype="object")
# Liste von Trainings und Testdaten erstellen. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Trainigsdaten erstellt")

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor 
#and one output tensor Quelle: https://keras.io/guides/sequential_model/
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Stochastischer Gradientenabstieg bietet hohe Genauigkeit Quelle: https://ieeexplore.ieee.org/abstract/document/9225395
# mit einen Nesterov Accelerated Gradient verwendet den letzetn und aktuellen Gradienten,
# um die Richtung des nächsten Abstiegs zu bestimmen (https://jlmelville.github.io/mize/nesterov.html#conclusions :Vergleich und Erklärung)
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# categorical_crossentropy sagt aus wie gut die Performanz eines Models ist. Sie wird genutzt wenn mehrere Labels verwendet werden 
#und benötigt ein One-Hot-Encoding, welches durch die Bags gegeben ist 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Training und speichern des Modells 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_hanover.h5', hist)
print("Modell erstellt")

