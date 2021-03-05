#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: CRF sequence tagging for Move Queries

# In[1]:


import os
import sys

from copy import deepcopy
from collections import Counter
from nltk.tag import CRFTagger

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import itertools
import collections

from matplotlib import pyplot as plt
import numpy as np

import pycrfsuite
import re
import unicodedata


# In[2]:


def get_raw_data_from_bio_file(fpath):
    """A simple function to read in from a one-word-per-line BIO
    (Beginning, Inside, Outside) tagged corpus, tab separated
    and each example sentence/text separated with a blank line.
    The data is already tokenized in a simple way.
    e.g.:
    
    O	a
    O	great
    O	lunch
    O	spot
    O	but
    B-Hours	open
    I-Hours	till
    I-Hours	2
    I-Hours	a
    I-Hours	m
    B-Restaurant_Name	passims
    I-Restaurant_Name	kitchen
    
    returns a list of lists of tuples of (word, tag) tuples
    """
    f = open(fpath)
    data = []  # the data, a list of lists of (word, tag) tuples
    current_sent = []  # data for current sentence/example
    for line in f:
        if line == "\n":  # each instance has a blank line separating it from next one
            # solution
            data.append(current_sent)
            current_sent = []
            continue
        line_data = line.strip("\n").split("\t")
        current_sent.append((line_data[1], line_data[0]))
    f.close()
    return data


# In[3]:


raw_training_data = get_raw_data_from_bio_file("engtrain.bio.txt") 


# In[4]:


# have a look at the first example
print(raw_training_data[0])


# In[5]:


print(len(raw_training_data), "instances")
print(sum([len(sent) for sent in raw_training_data]), "words")


# In[6]:

def preProcess(example):
    
    taggedTok = []
    preprocessExample = []
    """Function takes in list of (word, bio-tag) pairs, e.g.:
        [('what', 'O'), ('movies', 'O'), ('star', 'O'), ('bruce', 'B-ACTOR'), ('willis', 'I-ACTOR')]
    returns new (token, bio-tag) pairs with preprocessing applied to the words"""
    tokens, labels = zip(*example)
    posttagger = CRFTagger()
    posttagger.set_model_file("crf_pos.tagger")
    taggedTok = posttagger.tag(tokens)
        
    taggedTok  = list('@'.join(w) for w in taggedTok)
    preProcessExample = list(zip(taggedTok, labels))
        
    return preProcessExample
            
            
            #features = [self._feature_func(tokens, i) for i in range(len(tokens))]
            #tokenList.append(features, labels)




# In[7]:
def splitTraining(training_data):
    trainingSplit = training_data[:int(len(training_data)*0.8)]
    testSplit = training_data[int(len(training_data)*0.8):]
    return trainingSplit, testSplit


training_data = [preProcess(example) for example in raw_training_data]
#trainData, testData = splitTraining(training_data)


# In[8]:


# check the effect of pre-processing
print(training_data[0])


# In[9]:

_pattern = re.compile(r"\d")  # to recognize numbers/digits

# This is the 'out-of-the-box' get_features function from the nltk CRF tagger
def get_features(tokens, idx):
    """
    Extract basic features about this word including
         - Current Word
         - Is Capitalized ?
         - Has Punctuation ?
         - Has Number ?
         - Suffixes up to length 3
    Note that : we might include feature over previous word, next word ect.

    :return : a list which contains the features
    :rtype : list(str)

    """
    
    token, tag = tokens[idx].split("@")
    #if idx < len(tokens): 
        
    #nxtToken, nxtTag = tokens[idx+1].split("@")#split item into a list of token and tag
    
    
    #if idx > 0:
   # prevToken, prevTag = tokens[idx-1].split("@")
    
 

    feature_list = []

    if not token:
        return feature_list

    # Capitalization
    if token[0].isupper():
        feature_list.append("CAPITALIZATION")
    '''if nxtToken[0].isupper():
        feature_list.append("NEXT_CAPITALIZATION")'''
    #if prevToken[0].isupper():
       # feature_list.append("PREV_CAPITALIZATION")
        
    # Number
    if re.search(_pattern, token) is not None:
        feature_list.append("HAS_NUM")
    #if re.search(_pattern, prevToken) is not None:
      #  feature_list.append("PREV_HAS_NUM")
    

    # Punctuation
    punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
    if all(unicodedata.category(x) in punc_cat for x in token):
        feature_list.append("PUNCTUATION")

    # Suffix up to length 3
    if len(token) > 1:
        feature_list.append("PRE_" + token[0])
    if len(token) > 2:
        feature_list.append("PRE_" + token[:2])
    if len(token) > 3:
        feature_list.append("PRE_" + token[:3])
    if len(token) > 4:
        feature_list.append("PRE_" + token[:4])
        
     # Prefix up to length 3
    if len(token) > 1:
        feature_list.append("SUF_" + token[-1:])
    if len(token) > 2:
        feature_list.append("SUF_" + token[-2:])
    if len(token) > 3:
        feature_list.append("SUF_" + token[-3:])
    if len(token) > 4:
        feature_list.append("SUF_" + token[-4:])
        
    if idx == 0:
        feature_list.append("BOS")
    if idx == len(tokens)-1:
        feature_list.append("EOS") #End of sentence
        
    if token.istitle():
        feature_list.append("TITLE") 
   
        
        
    feature_list.append("WORD_" + token)
    
    feature_list.append("TAG_" + tag) #add the tag feature

        
    #NEXT WORD
    if idx < len(tokens)-1: 
        nxtToken, nxtTag = tokens[idx+1].split("@")
        
        if nxtToken[0].isupper():
            feature_list.append("NEXT_CAPITALIZATION")
            
        if re.search(_pattern, nxtToken) is not None:
            feature_list.append("NEXT_HAS_NUM")
        
        if len(nxtToken) > 1:
            feature_list.append("NEXT_PRE_" + nxtToken[0])
        if len(nxtToken) > 2:
            feature_list.append("NEXT_PRE_" + nxtToken[:2])
        if len(nxtToken) > 3:
            feature_list.append("NEXT_PRE_" + nxtToken[:3])
        if len(nxtToken) > 4:
            feature_list.append("NEXT_PRE_" + nxtToken[:4])  
            
        if len(nxtToken) > 1:
            feature_list.append("NEXTSUF_" + nxtToken[-1:])
        if len(nxtToken) > 2:
            feature_list.append("NEXTSUF_" + nxtToken[-2:])
        if len(nxtToken) > 3:
            feature_list.append("NEXTSUF_" + nxtToken[-3:])
        if len(nxtToken) > 4:
            feature_list.append("NEXTSUF_" + nxtToken[-4:])    
        

        if nxtToken.istitle():
            feature_list.append("NEXT_TITLE") 
            
        feature_list.append("NEXTWORD_" + nxtToken)
    
        feature_list.append("NEXTTAG_" + nxtTag) #add the tag feature
    #PREVIOUS WORD        

    if idx > 0:
        prevToken, prevTag = tokens[idx-1].split("@")
        
        if prevToken[0].isupper():
            feature_list.append("PREV_CAPITALIZATION")
        
        if re.search(_pattern, prevToken) is not None:
            feature_list.append("PREV_HAS_NUM")
        
        if len(prevToken) > 1:
            feature_list.append("PREV_PRE_" + prevToken[0])
        if len(prevToken) > 2:
            feature_list.append("PREV_PRE_" + prevToken[:2])
        if len(prevToken) > 3:
            feature_list.append("PREV_PRE_" + prevToken[:3])
        if len(prevToken) > 4:
            feature_list.append("PREV_PRE_" + prevToken[:4])  
        
        if len(prevToken) > 1:
            feature_list.append("PREVSUF_" + prevToken[-1:])
        if len(prevToken) > 2:
            feature_list.append("PREVSUF_" + prevToken[-2:])
        if len(prevToken) > 3:
            feature_list.append("PREVSUF_" + prevToken[-3:])
        if len(prevToken) > 4:
            feature_list.append("PREVSUF_" + prevToken[-4:])    
        


        if prevToken.istitle():
            feature_list.append("PREV_TITLE") 
        feature_list.append("NEXTWORD_" + prevToken)
    
        feature_list.append("NEXTTAG_" + prevTag)
    
     
    
    #print(feature_list)
    return feature_list

      

# In[10]:


# Train the CRF BIO-tag tagger
TAGGER_PATH = "crf_nlu.tagger"  # path to the tagger- it will save/access the model from here
ct = CRFTagger(feature_func=get_features)  # initialize tagger with get_features function

print("training tagger...")
ct.train(training_data, TAGGER_PATH)
print("done")


# In[11]:


# load tagger from saved file
ct = CRFTagger(feature_func=get_features)  # initialize tagger
ct.set_model_file(TAGGER_PATH)  # load model from file

# prepare the test data:
raw_test_data = get_raw_data_from_bio_file("engtest.bio.txt") 
test_data = [preProcess(example) for example in raw_test_data]
print(len(testData), "instances")
print(sum([len(sent) for sent in tesData]), "words")

# In[12]:



print("testing tagger...")
preds = []
y_test = []
sentList = []

for sent in test_data:
    sent_preds = [x[1] for x in ct.tag([s[0] for s in sent])]
    sent_true = [s[1] for s in sent]
    sentence = [s[0] for s in sent]
    preds.append(sent_preds) #CHANGE EXTEND() TO APPEND() TO BE USED TO FIND FALSE NEGATIVE AND POSITIVE AND VICE VERSA
    y_test.append(sent_true)  #CHANGE EXTEND() TO APPEND() TO BE USED TO FIND FALSE NEGATIVE AND POSITIVE AND VICE VERSA
    sentList.append(sentence) #append the list consists of only sentences 
    
"""
#================================== IDENTIFYING FALSE POSITIVE AND NEGATIVE ===============================
#PLEASE CHANGE THE EXTEND()FUNCTION IN THE PREVIOUS SECTION TO APPEND BEFORE RUNNING THE CODE BELOW


#Find the index of the sentence with target word


iPlot_idx = []

iSong_idx = []

bSong_idx = []
bChar_idx= []
bTrailer_idx = []


def findIdx(targetWord, targetList_idx):
    for i in y_test:
        if targetWord in i:
            targetList_idx.append(y_test.index(i))
            continue
            
            

findIdx('B-SONG', bSong_idx)
findIdx('I-PLOT', iPlot_idx)
findIdx('B-CHAR', bChar_idx)
findIdx('B-TRAILER', bTrailer_idx)



#comparing the two list
         
def compare_listsFP(A, B, targetWord, sudoList): #find false positives
    for idx, sub_lists in enumerate(zip(A,B)):
        for first, second in zip(sub_lists[0], sub_lists[1]):
            if first == targetWord and first != second and second != 'O': #not 'O' and not targetWord >> Wrong tag
                sudoList[idx].append(int(1)) #1 if falsePositive
            else:
                sudoList[idx].append(int(0)) #0 if else
                

def compare_listsFN(A, B, targetWord, sudoListFN): #find false negative
    for idx, sub_lists in enumerate(zip(A,B)):
        for first, second in zip(sub_lists[0], sub_lists[1]):
            if first == targetWord and second == 'O': #tagged as 0 instead of entity label
                sudoListFN[idx].append(int(1)) #1 if FN
            else:
                sudoListFN[idx].append(int(0)) #0 if else
                

def getSentences(sudoList, sentList, tagClass): #print out sentences of FP/FN given tagClass
    for idx, i in enumerate(sudoList): #here sudo list has the same index formation as the sentence list. 
        for v in i: #if there exist a 1 in the list, print the list 
            if v == 1:
                tagClass.append(sentList[idx])

def countDup(alist):
    blist = []
    alist =  [tuple(i) for i in alist] #convert element into tuples 
    blist = [item for item, count in collections.Counter(alist).items() if count > 1]
    
    return blist
                
                #==================================================================================
                
#Print false Positive for the 5 classes
                
iPlotFP = [[] for x in range(1955)]

bTrailerFP = [[] for x in range(1955)]
bSongFP = [[] for x in range(1955)]


compare_listsFP(y_test, preds, 'I-PLOT', iPlotFP)

compare_listsFP(y_test, preds, 'B-SONG', bSongFP)


bTrailerS = [] 
bSongS = [] 
iPlotS= [] 


getSentences(iPlotFP, sentList, iPlotS)

getSentences(bTrailerFP, sentList, bTrailerS)
getSentences(bSongFP , sentList, bSongS)


fpList = iPlotS+bTrailerS+bSongS#join all the five classes together
fpTup =  [tuple(i) for i in fpList] #convert element into tuples 
fpDup = [item for item, count in collections.Counter(fpTup).items() if count > 1] #returns 70 duplicates
fpList = set(map(tuple,fpTup))
#fpList.sort()
#fpList = list(fpList for fpList,_ in itertools.groupby(fpList))
fpList = list([' '.join(i)] for i in fpList) #join list of words into sentence 
for i in fpList:
    print(i)

                #==================================================================================

#printing the FN list from the five classes
bCharFn = [[] for x in range(1955)]
bSongFn = [[] for x in range(1955)]
bTrailerFn = [[] for x in range(1955)]
                

compare_listsFN(y_test, preds, 'B-CHARACTER', bCharFn)

compare_listsFN(y_test, preds, 'B-SONG', bSongFn)
compare_listsFN(y_test, preds, 'B-TRAILER', bTrailerFn)
bSongSn =[]
bCharSn = [] 

bTrailerSn = []
              
getSentences(bSongFn , sentList, bSongSn)
getSentences(bCharFn , sentList, bCharSn)
getSentences(bTrailerFn , sentList, bTrailerSn)

#Compare list for false negatives


fnList = bSongSn + bCharSn #join all the five classes together

fnTup =  [tuple(i) for i in fnList] 
fnList = set(map(tuple,fnTup))
#convert element into tuples 
fnDup = [item for item, count in collections.Counter(fnTup).items() if count > 1] #returns duplicates
fnList = set(map(tuple,fnTup))
#fpList.sort()
#fpList = list(fpList for fpList,_ in itertools.groupby(fpList))
fnList = list([' '.join(i)] for i in fnList) #join list of words into sentence 
for i in fnList:
    print(i)

    
print("done")
"""

# In[13]:


# Output the classification report (which you should save each time for comparing your models)
print(classification_report(y_test, preds))


# In[14]:


def confusion_matrix_heatmap(y_test, preds):
    """Function to plot a confusion matrix"""
    labels = list(set(y_test))   # get the labels in the y_test
    # print(labels)
    cm = confusion_matrix(y_test, preds, labels)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels( labels, rotation=45)
    ax.set_yticklabels( labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    #fig.tight_layout()
    plt.show()


# In[15]:


confusion_matrix_heatmap(y_test, preds)


# # 1. Split the training data into 80% training, 20% development set (5 marks)
# Split the training data (`training_data`) into two lists: one split of the first 80% of the instances of `training_data`, which you will use for training your CRF, and the remaining 20% for testing. Once you've done this re-run the above code such that the tagger is trained on the 80% split and tested on the 20% split, and you obtain the classification report output and confusion heatmap output for the results of testing. Do not use the test data as it is above for testing/viewing results for now. Record the results by saving the classification report output as a string somewhere in the notebook for future reference as you go through.
# 

# # 2. Error analysis 1: False positives (10 marks)
# 
# Performing error analyses is a key part of improving your NLP applications. For the 5 classes which have the lowest precision, according to the results table from your 20% development data, print out all the sentences where there is a false positive for that class (i.e. the label is predicted in the predicted label for a given word by the tagger, but this is not present in the corresponding ground truth label for that word). HINT: This may most easily be achieved by editing the code above beginning `print("testing tagger...")` and ending `print("done")`. Have a look at these errors closely, and think about which features could be added to reduce the number of these errors.
# 

# # 3. Error analysis 2: False negatives (10 marks)
# 
# For the 5 classes which have the lowest recall, according to the results table from your 20% development data,, print out all the sentences where there is a false negative for that label (i.e. the label is present in the ground truth label for a given word, but that label is not predicted for that word by the tagger). HINT: This may most easily be achieved by editing the code above beginning `print("testing tagger...")` and ending `print("done")`. Have a look at these errors closely, and think about which features could be added to reduce the number of these errors.
# 

# # 4. Using POS tags as features (15 marks)
# Use the CRF part-of-speech (POS) tagger as shown below to add POS tags to the words in the training data. Do this by altering the `preProcess` function above. Note the CRF tagger only takes strings as input so you will have to concatenate the word and POS tag together (with a special symbol, e.g. @), and you will also have to then split on this special symbol in the feature extraction function `get_features` to get the word and POS tag - modify that function so it uses the POS tag in addition to the word (currently using the word only is achieved by `feature_list.append("WORD_" + token)`. Re-run the training and testing code on your 80%/20% training/dev split from question 1 and record the results from the classification report as text in this file for comparison of the accuracy metrics against not using POS tags- try to see any improvemements across the classes.

# In[16]:


# a postagger for use in exercises
posttagger = CRFTagger()
posttagger.set_model_file("crf_pos.tagger")
# example use:
words = ["john", "likes", "mary", "and", "bill"]
print(posttagger.tag(words))


# # 5. Feature experimentation for optimal macro average (20 marks).
# Experiment with different features by further adjusting the `get_features` function, and modifying it to get the best results in terms of `macro average f-score` (i.e. average f-score across all classes) on your 20% development data. Iteratively try different functions, briefly describe the method and record the results in the classification report format. You could try more suffixes/prefixes of the curret word than those currently extracted, you could use windows of the next and previous tokens (of different sizes, e.g. the previous/next N words/tags). As you try different feature functions, use the techniques you used in Q1 and Q2 to see the kind of errors you are getting for lower performing classes, in addition to the confusion matrix over classes. Leave the `get_features` functions in the state you used to get the highest `macro average f-score` on your 20% development set, then re-train the model on ALL the training data and print the classification report for the original test data (i.e. from the test file `engtest.bio.txt`) as your final piece of code.

# In[ ]:




