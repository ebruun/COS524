import pandas as pd
import numpy as np
import regex as re
import string
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) - {'all'}

import preprocessing as pre

print ("Imports are all good!")






# Retrieves and preprocesses the training dataset
path = "./data/train.csv" # Path to train.csv
train_df = pd.read_csv(path)
train_df.fillna("", inplace=True) # fills any NaN values with empty strings
train_df = pre.preprocess_df(train_df)
train_df.head(5)

# Uses a CountVectorizer to construct bag-of-words matrix
vectorizer = CountVectorizer() # Add a comment about the max_features & ngram_range parameters

# train_vocab is an 2d array of the vocab from the training dataset 
train_vocab = vectorizer.fit_transform(train_df['text']).toarray()

# train_vocab_df is a dataframe where the element ij is the number of times word j occurred in Tweet i
train_vocab_df = pd.DataFrame(train_vocab, columns=vectorizer.get_feature_names())
train_labels = train_df['BLM']

train_df.head()
#print (train_labels.count() + 1)

#train_vocab_df.head()
#print (train_vocab_df.count() + 1)

#train_labels.head()
#print (train_labels.count() + 1)

# Retrieves and preprocesses the training dataset
path = "./data/train.csv" # Path to train.csv
train_df = pd.read_csv(path)
train_df.fillna("", inplace=True) # fills any NaN values with empty strings
train_df = preprocess_df(train_df)
train_df.head(5)

# Uses a CountVectorizer to construct bag-of-words matrix
vectorizer = CountVectorizer() # Add a comment about the max_features & ngram_range parameters

# train_vocab is an 2d array of the vocab from the training dataset 
train_vocab = vectorizer.fit_transform(train_df['text']).toarray()

# train_vocab_df is a dataframe where the element ij is the number of times word j occurred in Tweet i
train_vocab_df = pd.DataFrame(train_vocab, columns=vectorizer.get_feature_names())
train_labels = train_df['BLM']

train_df.head()
#print (train_labels.count() + 1)

#train_vocab_df.head()
#print (train_vocab_df.count() + 1)

#train_labels.head()
#print (train_labels.count() + 1)

# Retrieves and preprocesses the test dataset
path = "./test.csv" # Path to Test_dataset.csv
test_df = pd.read_csv(path)
test_df.fillna("", inplace=True) # fills any NaN values with empty strings
test_df = preprocess_df(test_df)
test_df.head(5)


# Uses the vocab from the training dataset to vectorize the test dataset
test_vocab = vectorizer.transform(test_df['text']).toarray()

# test_vocab_df is a dataframe where the element ij is the number of times word j
# occurred in Tweet i
test_vocab_df = pd.DataFrame(test_vocab, columns=vectorizer.get_feature_names())
test_labels = test_df['BLM']


test_vocab_df.head()
test_labels.head()

print(f"Number of neither Tweets in training: {len(train_df[train_df['BLM'] == 'neither'])}")
print(f"Number of positive Tweets in training: {len(train_df[train_df['BLM'] == 'positive'])}")
print(f"Number of negative Tweets in training: {len(train_df[train_df['BLM'] == 'negative'])}")

print(f"Number of neither Tweets in test: {len(test_df[test_df['BLM'] == 'neither'])}")
print(f"Number of positive Tweets in test: {len(test_df[test_df['BLM'] == 'positive'])}")
print(f"Number of negative Tweets in test: {len(test_df[test_df['BLM'] == 'negative'])}")