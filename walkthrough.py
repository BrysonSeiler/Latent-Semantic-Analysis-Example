"""
This script is a walked-through example for 
the process of Latent Semantic Analysis.

@author: Bryson Seiler
"""


import re
import numpy as np
import pandas as pd
from scipy import linalg, spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)

from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#Here are the documents that we will be using:
corpus = ['Cats are good pets. They are clean and not noisy. Dogs are not as clean as cats, but are still great pets and companions.',
          'Cats and dogs have always been great household pets. While cats prefer to be inside. Dogs love to go on walks.',
          'Italy is my favorite country. There is so much to see, so much to do. If you are looking to travel to a different country, you should travel to Italy.',
          'There are 195 countries in the world today, many of which you should travel to. In particular, Italy is an excellent travel destination with lots to see.']


#Set stop words to english
stop_words = set(stopwords.words('english'))

filtered_document = []
filtered_corpus = []


'''
We will first remove all special characters, 
excess whitespace, and stopwords from the documents.
'''

for document in corpus:

    #Clean up the document by removing numbers and special characters and excess whitespace
    clean_document = " ".join(re.sub(r"[^A-Za-z \—]+", " ", document).split())

    #Tokenize document
    document_tokens = word_tokenize(clean_document)

    #Remove stopwords
    for word in document_tokens:
        if word not in stop_words:
            filtered_document.append(word)

    filtered_corpus.append(' '.join(filtered_document))

    filtered_document = []


'''
Now we will use sklearn to convert our filtered corpus
into a document-term frequency matrix (counts matrix).
'''

vectorizer = CountVectorizer()

counts_matrix = vectorizer.fit_transform(filtered_corpus)

#Get feature names
feature_names = vectorizer.get_feature_names()

#Put the counts matrix into a data frame
count_matrix_df = pd.DataFrame(counts_matrix.toarray(), columns=feature_names)
count_matrix_df.index = ['Document 1','Document 2','Document 3','Document 4']

print("Word frequency matrix: \n", count_matrix_df)

'''
The next step is to convert our term-document frequency
matrix into a term frequency inverse document frequency
matrix (tfidf).
'''

transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(counts_matrix)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.index = ['D1','D2','D3','D4']

print("Tfidf matrix shape: ", tfidf_df.shape)
print("Tfidf matrix: \n", tfidf_df)

