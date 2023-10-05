#!/usr/bin/env python
# coding: utf-8

# ## CS 6120: Natural Language Processing - Prof. Ahmad Uzair
# 
# ### Assignment 1: Naive Bayes
# ### Total Points: 100 points
# 
# You will be dealing with movie review data that includes both positive and negative reviews in this assignment. You will use Sentiment Analysis to assess if a given review is positive or negative using the provided dataset.
# 
# Therefore, we will make use of Naive Bayes algorithm to perform sentiment analysis on the movie review dataset.
# 
# ## Importing the Libraries

# In[2]:


import numpy as np
import math
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
打法




# TASK CELL
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = nltk.PorterStemmer()#stemming

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def toLowerCase(text):
    return text.lower() #changes all upper case alphabet to lower case

import string
string.punctuation # checking punctuations
def removePunctuation(text):
    return "".join([char for char in text if char not in string.punctuation])#removePunctuation
nltk.download('stopwords')

import re
def removeURLs(text):
    
    text = re.sub(r"http\S+", "", text) # replaces URLs starting with http 
    text = re.sub(r"www.\S+", "", text) # replaces URLs starting with wwe
    return text

def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 

    '''
    text1 = toLowerCase(review)#toLowerCase
    text2 = removeURLs(text1)#removeURLs
    text = removePunctuation(text2)#removePunctuation
    words = text.split()
    r = []
    for w in words:
        if w not in stopwords.words("english"):
            r.append(ps.stem(w))
    review_cleaned=' '.join(r)


    return review_cleaned


import math# for calculate



# In[55]:


# TASK 4 CELL

def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    
      # process the review to get a list of words
    word_l = clean_review(review).split()

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob = logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood.keys():
            # add the log likelihood of that word to the probability
            total_prob = total_prob + float(loglikelihood[word])
    if total_prob >0:
        re = 1
    else:
        re=0


    return re,total_prob


para = pd.read_csv("parameter2.csv", sep = ',', encoding = 'latin-1', usecols = lambda col: col not in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
logprior = float(para['logprior'])
a = input("input text:")
if a == 'X':
    print(' Program quit')
else:
    re,total_prob = naive_bayes_predict(a, logprior, para)
    print('final classification decision',re)
    print('Prob',total_prob)

