"""
This script is a walked-through example for 
the process of Latent Semantic Analysis.

@author: Bryson Seiler
"""


import re
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import linalg, spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)

from sklearn.utils.extmath import randomized_svd

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

#print("Word frequency matrix: \n", count_matrix_df)


'''
The next step is to convert our term-document frequency
matrix into a term frequency inverse document frequency
matrix (tfidf).
'''

transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(counts_matrix)

tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_matrix_df.index = ['D1','D2','D3','D4']

#print("Tfidf matrix: \n", tfidf_matrix_df)


'''
Now that we have the tfidf matrix, we project the matrix
into semantic space through the use of singular-value-decomposition.
'''

#Transpose tfidf matrix so that the columns are the documents
tfidf_transpose = tfidf_matrix.transpose()

#Find the singular value decomposition of the transposed tfidf matrix
U, Sigma, VT = randomized_svd(tfidf_transpose, 
                              n_components=4,
                              n_iter=10,
                              random_state=None)


'''
Since randomized_svd produces the singular values of the transposed
tfidf matrix, we need a helper method to produce a diagonal matrix
that consists of the singular values.
'''

def singular_value_matrix(s,m,n):
    matrix = []
    temp_matrix = []
    for row in range(0,m):
        for col in range(0,n):
            if row == col:
                temp_matrix.append(s[row])
            else:
                temp_matrix.append(0)
        matrix.append(temp_matrix)
        temp_matrix = []
    return matrix

sigma_matrix = np.asarray(singular_value_matrix(Sigma, len(Sigma), len(Sigma)))

'''
Now that we have the singular-value-decomposition of the transposed
tfidf matrix, the final step of latent semantic analysis requires
the computation of V*S, which projects the transposed tfidf matrix
into semantic space.
'''

projection_matrix = np.matmul(VT.transpose(), sigma_matrix)

projection_matrix_df = pd.DataFrame(projection_matrix, columns=['f1','f2','f3','f4'])
projection_matrix_df.index = ['D1','D2','D3','D4']

print("Semantic Space (VS): \n", projection_matrix_df)
