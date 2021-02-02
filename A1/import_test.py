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

print ("Imports are all good!")