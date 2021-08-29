import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow
import tflearn
import random 
import json

with open("intents.json") as file:
    data = json.load(file)

words = [] 
classes = []
docs = [] 

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in classes:
        classes.append(intent["tag"])
