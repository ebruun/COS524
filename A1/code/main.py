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

print("new stuff i'm adding")

print("I am a master of GITHUB")

print(practice github)

print("this work willget lost")
print("Edvard making changes on Darshan branch")
