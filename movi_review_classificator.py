# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:48:51 2021

@author: ala_j
"""
from tensorflow.keras.datasets import imdb
import numpy as np


def vectorize_sequences(sequence, dimension=10000):
    """
    Transform reviews list from list to One Hot encoding and 
    return tensor of shape (, dimension)

    Parameters
    ----------
    sequence : list of integers
        data extracted from IMDB dataset.
    dimension : integer, optional
        dimension of the input data. The default is 10000.

    Returns
    -------
    tensor of shape (, dimension)

    """
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1.
        
    return results
    

# Loading data set splitted in train and test
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# Setting num_words=10000 take the top 10.000 most frequent words
# no word index will exceed 9.999
print(f'MAX Index: {max([max(sequence) for sequence in train_data])}')
# Decode review to English
word_index = imdb.get_word_index()
reversed_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
    )
decoded_review = ' '.join(
    [reversed_word_index.get(i - 3, '?') for i in train_data[0]]
    )
print(f'Review decoding: {decoded_review}')

# PREPARING THE DATA
# Apply One-Hot encoding for train and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# Vectorize labels
y_train = np.asarray(train_labels).dtype('float32')
y_test = np.asarray(test_labels).dtype('float32')

