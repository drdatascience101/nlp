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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import umap

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

    stems.append(stemmed)

# Rejoin content
joined = []
    
for i in range(len(stems)):
    
    t = ' '.join(stems[i])
    joined.append(t)
    
'''
    Modeling:
        
        - TF-IDF
        - SVD
        - Visualization
'''

# TF-IDF
vec = TfidfVectorizer(max_features = 1000, max_df = 0.5, smooth_idf = True)

x = vec.fit_transform(joined)

# SVD
svd = TruncatedSVD(n_components = 20, algorithm = 'randomized', n_iter = 100, random_state = 100)

svd.fit(x)

# Labels
terms = vec.get_feature_names()

for i, comp in enumerate(svd.components_):
    
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key = lambda x:x[1], reverse = True)[:7]
    
    print("Topic "+ str(i) + ": ", [t[0] for t in sorted_terms])
    
# Visualize
topics = svd.fit_transform(x)

embedding = umap.UMAP(n_neighbors = 150, min_dist = 0.5, random_state = 100).fit_transform(topics)

plt.figure(figsize = (20, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c = df.target, s = 10, edgecolor = 'none')
plt.show()
plt.savefig('toxicity_lda.png')
