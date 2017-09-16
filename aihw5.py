import numpy as np
import sklearn
import sklearn.datasets
import re

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

from sklearn.metrics import confusion_matrix

#loads data from all files, combines it, splits it into a training and testing set, and stores them in new files
def loadAllData():
    textdata = []
    amazon = open("amazon_cells_labelled.txt")
    imdb = open("imdb_labelled.txt")
    yelp = open("yelp_labelled.txt")
    for line in amazon.readlines():
        textdata.append(line)
    for line in imdb.readlines():
        textdata.append(line)
    for line in yelp.readlines():
        textdata.append(line)
    shuffle(textdata)
    newdata = trainingSplit(textdata)
    trainingdata = open("trainingdata.txt",'w')
    testdata = open("testdata.txt",'w')
    for item in newdata[0]:
        trainingdata.write(item)
    for item in newdata[1]:
        testdata.write(item)
    amazon.close()
    imdb.close()
    yelp.close()
    trainingdata.close()
    testdata.close()

#converts data from a file into a more manageable form
def loadListData(filename):
    x = open(filename)
    textlist = []
    for line in x.readlines():
        splittext = line.split("\t")
        splittext[0] = splittext[0].replace("."," ")
        splittext[0] = splittext[0].replace("!"," ")
        splittext[0] = splittext[0].replace("?"," ")
        splittext[0] = splittext[0].replace(",", " ")
        splittext[0] = splittext[0].replace("\"", " ")
        splittext[0] = splittext[0].replace(":", " ")
        splittext[0] = splittext[0].replace(")", " ")
        splittext[0] = splittext[0].replace("(", " ")
        splittext[0] = splittext[0].replace("-", " ")
        splittext[0] = splittext[0].replace("[", " ")
        splittext[0] = splittext[0].replace("]", " ")
        splittext[0] = splittext[0].replace("#", " ")
        splittext[0] = splittext[0].replace("*", " ")
        splittext[0] = splittext[0].replace("$", " ")
        splittext[0] = splittext[0].lower()
        splitsentence = splittext[0].split()
        textlist.append((splitsentence,splittext[1].split("\n")[0]))
    x.close()
    return textlist

#splits the attributes and class into two separate lists (after shuffling data)
def dataSplit(vals):
    featurevector = []
    classes = []
    for i in range(len(vals)):
        featurevector.append(vals[i][0])
        classes.append(vals[i][1])
    return(featurevector,classes)

#computes decision tree and predicts test data using unigram model
def treeLearn(trainingdata,testdata):
    treeclassifier = DecisionTreeClassifier(criterion='entropy')
    #Xtrain = transformTrainingData(trainingdata[0])
    #Xtest = transformTestData(testdata[0])

    vectorizer = CountVectorizer(min_df=1)

    list1 = []
    for x in trainingdata[0]:
        strval = ""
        for y in x:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list1.append(strval)
    Xtrain = vectorizer.fit_transform(list1)

    list2 = []
    for xt in testdata[0]:
        strval = ""
        for y in xt:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list2.append(strval)
    Xtest = vectorizer.transform(list2)
    
    treeclassifier.fit(Xtrain.toarray(),trainingdata[1])
    print("Success Rate:")
    print(treeclassifier.score(Xtest.toarray(),testdata[1]))
    print("Confusion Matrix:")
    print(confusion_matrix(treeclassifier.predict(Xtest.toarray()),testdata[1]))

#computes decision tree and predicts test data using bigram model
def treeLearnBigram(trainingdata,testdata):
    treeclassifier = DecisionTreeClassifier(criterion='entropy')
    #Xtrain = transformTrainingData(trainingdata[0])
    #Xtest = transformTestData(testdata[0])

    vectorizer = CountVectorizer(min_df=1,ngram_range=(2,2))

    list1 = []
    for x in trainingdata[0]:
        strval = ""
        for y in x:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list1.append(strval)
    Xtrain = vectorizer.fit_transform(list1)

    list2 = []
    for xt in testdata[0]:
        strval = ""
        for y in xt:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list2.append(strval)
    Xtest = vectorizer.transform(list2)
    
    treeclassifier.fit(Xtrain.toarray(),trainingdata[1])
    print("Success Rate:")
    print(treeclassifier.score(Xtest.toarray(),testdata[1]))
    print("Confusion Matrix:")
    print(confusion_matrix(treeclassifier.predict(Xtest.toarray()),testdata[1]))
    #print(treeclassifier.predict(Xtest[742]))
    #print(testdata[1][742])
    #print(testdata[0][742])

#computes decision tree and predicts test data using trigram model
def treeLearnTrigram(trainingdata,testdata):
    treeclassifier = DecisionTreeClassifier(criterion='entropy')
    #Xtrain = transformTrainingData(trainingdata[0])
    #Xtest = transformTestData(testdata[0])

    vectorizer = CountVectorizer(min_df=1,ngram_range=(3,3))

    list1 = []
    for x in trainingdata[0]:
        strval = ""
        for y in x:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list1.append(strval)
    Xtrain = vectorizer.fit_transform(list1)

    list2 = []
    for xt in testdata[0]:
        strval = ""
        for y in xt:
            strval = strval + y + ","
        strval = strval[0:(len(strval)-1)]
        list2.append(strval)
    Xtest = vectorizer.transform(list2)
    
    treeclassifier.fit(Xtrain.toarray(),trainingdata[1])
    print("Success Rate:")
    print(treeclassifier.score(Xtest.toarray(),testdata[1]))
    print("Confusion Matrix:")
    print(confusion_matrix(treeclassifier.predict(Xtest.toarray()),testdata[1]))
    #print(treeclassifier.predict(Xtest[742]))
    #print(testdata[1][742])
    #print(testdata[0][742])

#code to call the program
trainingdata = loadListData("trainingdata.txt")
testdata = loadListData("testdata.txt")

training = dataSplit(trainingdata)
test = dataSplit(testdata)
treeLearn(training,test) #calls main functions
treeLearnBigram(training,test)
treeLearnTrigram(training,test)


#call this to load all the data from the 3 files, and split it into training and testing sets in separate files
##loadAllData()



