#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import string
 



# In[2]:


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            if line[0] == "DOC_ID":  # this skip the entire first row as the condition is met and it iliterate the next row. 
                continue
            (Id, Rating, verified_Purchase, product_Category, Text, Label) = parseReview(line)
            rawData.append((Id, Rating, verified_Purchase, product_Category, Text, Label))
            #preprocessedData.append((Id, preProcess(Text), Label))
        
def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Rating, verified_Purchase, product_Category, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)),Label))
    for (_, Rating, verified_Purchase, product_Category, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)),Label))


# # Question 1

# In[3]:


# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    Id = reviewLine[0]
    Text = reviewLine[8]
    Label = reviewLine[1]
    Rating = reviewLine[2]
    verified_Purchase = reviewLine[3],
    product_Category = reviewLine[4]
    
    tuple = ((int(Id)), Rating, verified_Purchase, product_Category, Text, fakeLabel if Label=='__label1__' else realLabel)
    # DESCRIBE YOUR METHOD IN WORDS
    '''Simply converting the first, ninth, and secvond column of every row into the target variable.
    '''
    
    return tuple


# In[4]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

table = str.maketrans({key: None for key in string.punctuation}) #returns a mapping table for translation using unicode ordinal
# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens
    '''text.lower() #lowercase everything #ORIGINAL FUNCTION
    tokens = word_tokenize(text)'''
    
    # Stemming and lemmatisation 
    lemmatizer = WordNetLemmatizer()
    new_tokens=[]
    lemmatised_tokens = []
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        if w not in stop_words:
            lemmatised_tokens.append(lemmatizer.lemmatize(w.lower()))
        new_tokens = [' '.join(l) for l in nltk.bigrams(lemmatised_tokens)] + lemmatised_tokens
    return new_tokens

    '''split the text into tokens of words based on white space'''
    '''Can also use the split(" ") function since the document uses tabs to seperate the words'''
    
    # DESCRIBE YOUR METHOD IN WORDS
    #return tokens


# # Question 2

# In[5]:


featureDict = {} # A global dictionary of features

def toFeatureVector(Rating, verified_Purchase, product_Category, tokens): #
    
    localDict = {} #previously wordfreq = {}
    
#Rating

    #print(Rating)
    featureDict["R"] = 1   
    localDict["R"] = Rating

#Verified_Purchase
  
    featureDict["VP"] = 1
            
    if verified_Purchase == "N":
        localDict["VP"] = 0
    else:
        localDict["VP"] = 1

#Product_Category

    
    if product_Category not in featureDict:
        featureDict[product_Category] = 1
    else:
        featureDict[product_Category] +=1
            
    if product_Category not in localDict:
        localDict[product_Category] = 1
    else:
        localDict[product_Category] += 1
            
#Text
        
    for token in tokens:
        if token not in localDict.keys():
            localDict[token] = 1 #add keys to dict if does not exist before
        else:
            localDict[token] += 1 #if the key already exist, increment the value.
            
        if token not in featureDict:
            featureDict[token] = 1
        
        else:
            featureDict[token] += 1
            
    return localDict#wordfreq
'''
CAN ALSO USE PYTHON COLLECTIONS: COUNTERS FUNCTION --> #dict(Counter())

'''
    # DESCRIBE YOUR METHOD IN WORDS
    

# In[6]:


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC(C=0.01))]) #allow misclassification by expanding the margin 
    return SklearnClassifier(pipeline).train(trainData)


# # Question 3

# In[7]:


def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = int(len(dataset)/folds)
    
    # DESCRIBE YOUR METHOD IN WORDS
    for i in range(0,len(dataset),foldSize):
        test_fold = dataset[i:i+foldSize] # test fold
        training_fold = dataset[0:i] + dataset[i+foldSize:] #our training fold excluding testfold
        classifier = trainClassifier(training_fold) #train on our training fold bycalling the trainClassifer function
        y_pred = predictLabels(test_fold, classifier) #predict the label on our test fold
        y = list(map(lambda x: x[1], test_fold)) #convert actual y value of test_fold into a list (function returns label)
        values = list(precision_recall_fscore_support(y, y_pred, average='weighted')) #calculate precision, recall, and fscore
        values[3] = accuracy_score(y, y_pred) * 100 #add an accuracy values onto the list
        cv_results.append(tuple(values))
        
    return cv_results
  
'''
  Accuracy = TP+TN/TP+FP+FN+TN
  Precision = TP/TP+FP
  Recall = TP/TP+FN
  F1 score - F1 Score is the weighted average of Precision and Recall
  
'''
    


# In[8]:


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# In[9]:


# MAIN

# loading reviews
# initialize global lists that will be appended to by the methods below
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
trainData = []        # the pre-processed training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the pre-processed test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'


# Do the actual stuff (i.e. call the functions we've made)
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath) 

# We split the raw dataset into a set of training data and a set of test data (80/20)
# You do the cross validation on the 80% (training data)
# We print the number of training samples and the number of features before the split
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features after the split
print("After split, %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

# QUESTION 3 - Make sure there is a function call here to the

# We print the number of training samples and the number of features
cvResults = crossValidate(trainData, 10)
display(cvResults)
print("Done!")

# crossValidate function on the training set to get your results


# # Evaluate on test set

# In[10]:


# Finally, check the accuracy of your classifier by training on all the tranin data
# and testing on the test set
# Will only work once all functions are complete
functions_complete = True  # set to True once you're happy with your methods for cross val
if functions_complete:
    print(testData[0])   # have a look at the first test data instance
    classifier = trainClassifier(trainData)  # train the classifier
    testTrue = [t[1] for t in testData]   # get the ground-truth labels from the data
    testPred = predictLabels(testData, classifier)  # classify the test data to get predicted labels
    finalScores = precision_recall_fscore_support(testTrue, testPred, average='weighted') # evaluate
    print("Done training!")
    print("Precision: %f\nRecall: %f\nF Score:%f" % finalScores[:3])


# # Questions 4 and 5
# Once you're happy with your functions for Questions 1 to 3, it's advisable you make a copy of this notebook to make a new notebook, and then within it adapt and improve all three functions in the ways asked for in questions 4 and 5.

# In[ ]:




