#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:09:17 2019

@author: zxs
"""

# Import the required libraries
import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
 
# Set the working directory
wd = '/Users/zxs/Documents/code/kaggle/sentiment/'
os.chdir(wd)

# Load the data
df = pd.read_csv('train.csv.zip', compression = 'zip')
df = df.sample(frac = .1, random_state = 100)

'''
    Text Processing:
        
        - lower case
        - remove punctuation
        - tokenize
        - remove stop words
        - stem the remaining words
'''

# Filter stopwords
stop_words = set(stopwords.words('english'))

# Stem the words
ps = PorterStemmer()

# Fix the casing
text = [i.lower() for i in df['comment_text']]

# Remove punctuation
no_punct = [i.translate(str.maketrans('', '', string.punctuation)) for i in text]

# Tokenize the words
tokens = [word_tokenize(x) for x in no_punct]

# Remove stop words
no_stops = []

for i in tokens:
    
    no_stops.append([x for x in i if x not in stop_words])
    
# Stemming
stems = []

for i in no_stops:
    
    stemmed = [ps.stem(x) for x in i]

    joined = [''.join(x) for x in stemmed]

    stems.append(joined)

'''
    Modeling:
        
        - Term Document Matrix
        - LDA
'''

# Create a dictionary for the corpus and give each unique term an index
dictionary = corpora.Dictionary(stems)

# Term document matrix
term_mat = [dictionary.doc2bow(comment) for comment in stems]

# LDA
lda = gensim.models.ldamodel.LdaModel

# Model
ldamod = lda(term_mat, num_topics = 3, id2word = dictionary, passes = 50)

# Examine results
top_3 = [i for i in ldamod.print_topics(num_topics = 3, num_words = 3)]
