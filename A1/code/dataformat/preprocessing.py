#This class is based on code provided by TAs in the assignment 

import regex as re
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Preprocessor():

	def __init__(self):
		print("PREPROCESSOR OBJECT CREATED")

		self.stop_words = set(stopwords.words('english')) - {'all'}
	

	# Preprocesses the text of the Tweets in the df and returns the df
	# By default, this removes the Tweets with the "neither" label
	def preprocess_df(self,df, remove_neither=True):
		print("--preprocessing data...")
		idx = "text"
		length = len(df[idx])

		for ii in range(5):
			tweet = str(df[idx][ii])
			df.loc[ii, idx] = self.preprocess_text(tweet)

		if (remove_neither):
			new_df = df[df.BLM != 'neither']
			return new_df
		else:
			return df

		
	# Preprocesses the tweets text
	# This function is based on code from:
	# https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python
	def preprocess_text(self,tweet):
		# Changes emojis to words
		tweet = emoji.demojize(tweet,  delimiters=(' ', ' '))
		# Removes 'RT' from tweet
		tweet = re.sub(r'RT[\s]+', '', tweet)
		# Removes capitalization
		tweet = tweet.lower()
		# Removes urls & user mentions from tweet
		tweet = re.sub(r"http\S+|www\S+|https\S+|\@\w+", ' ', tweet, flags=re.MULTILINE)
		# Removes punctuation
		tweet = re.sub(r'\p{P}+', '', tweet)
		# Removes stopwords
		tokens = [w for w in word_tokenize(tweet) if not w in self.stop_words]
		# Perfoms lemmatization on tokens
		lemmatizer = WordNetLemmatizer()
		lemma_words = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens]
		return " ".join(lemma_words)

	# Gets the part of speech tag of word for lemmatization
	# This function is based on code from:
	#   https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
	def get_wordnet_pos(self,word):
		tag = nltk.pos_tag([word])[0][1][0].upper()
		tag_dict = {"J": wordnet.ADJ,
					"N": wordnet.NOUN,
					"V": wordnet.VERB,
					"R": wordnet.ADV}
		return tag_dict.get(tag, wordnet.NOUN)