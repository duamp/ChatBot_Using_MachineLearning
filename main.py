import nltk #natural language tool kit for tokenization(hide meaningful data)
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow
import tflearn
import random 
import json

with open("intents.json") as file: #open file
    data = json.load(file)

words = [] #broken up words
classes = []
docs_x = [] #list of patterns 
docs_y = [] #list of tags 

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"]) 

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words] #what is stemmer.stem
words = sorted(list(set(words)))

classes = sorted(classes) #sorted tags 

training = [] 
output = []

out_empty = [0 for _ in range(len(classes))] #?

for x, doc in enumerate(docs_x): #?
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:] #?
    output_row[classes.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

traning = numpy.array(training)
output = np.array(output)


