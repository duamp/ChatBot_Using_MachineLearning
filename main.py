import nltk #natural language tool kit for tokenization(hide meaningful data)
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow as tf
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
        docs_x.append(wrds)
        docs_y.append(intent["tag"]) 

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #what is stemmer.stem
words = sorted(list(set(words)))

classes = sorted(classes) #sorted tags 

training = [] 
output = []

out_empty = [0 for _ in range(len(classes))] #?

for x, doc in enumerate(docs_x): # x = position & docs_x = text
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:] #[:] = all elements of the array
    output_row[classes.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

traning = numpy.array(training)
output = numpy.array(output)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #input data (length of training data)
net = tflearn.fully_connected(net,8) # hidden layer (8 nuerons, fully connected)
net = tflearn.fully_connected(net,8) # hidden layer (8 nuerons, fully connected)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax") # output layer connects nueron with classes
net = tflearn.regression(net) #

model = tflearn.DNN(net) #DNN = type of nueral network 

model.fit(training,output,n_epoch=1000,batch_size = 8,show_metric=True) #start passing training data, n_epoch = how many times it sees data
model.save("model.tflearn")

