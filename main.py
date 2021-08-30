import nltk #natural language tool kit for tokenization(hide meaningful data)
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tensorflow as tf
import tflearn
import random 
import json
import pickle

with open("intents.json") as file: #open file
    data = json.load(file)

try:
    with open ("data.pickle","rb") as f:
        words,classes,training, output = pickle.load(f)

except:
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

        with open ("data.pickle","wb") as f:
            pickle.dump((words,classes,training, output),f)

    traning = numpy.array(training)
    output = numpy.array(output)

tf.compat.v1.reset_default_graph()

try:
    model.load("model.tflearn")
except:
    net = tflearn.input_data(shape=[None, len(training[0])]) #input data (length of training data)
    net = tflearn.fully_connected(net,8) # hidden layer (8 nuerons, fully connected)
    net = tflearn.fully_connected(net,8) # hidden layer (8 nuerons, fully connected)
    net = tflearn.fully_connected(net,len(output[0]),activation="softmax") # output layer connects nueron with classes
    net = tflearn.regression(net) #

    model = tflearn.DNN(net) #DNN = type of nueral network 

    model.fit(training,output,n_epoch=1000,batch_size = 8,show_metric=True) #start passing training data, n_epoch = how many times it sees data
    model.save("model.tflearn")

def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You ")
        if inp.lower() == "quit":
            break
        results = model.predict([bagOfWords(inp,words)])
        results_index = numpy.argmax(results)
        tag = classes[results_index]
        
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))

chat()