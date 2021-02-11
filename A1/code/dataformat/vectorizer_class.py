from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer():
	def __init__(self,which_vectorizer, ngram):
		print("\nVECTORIZER OBJECT CREATED")

		self.name = which_vectorizer
		self.ngram = ngram
		self.vectorizer = self.create_vectorizer()

	def create_vectorizer(self):
		if self.name == "count":
			print("--count vectorizer")
			return CountVectorizer(ngram_range = self.ngram)
		if self.name == "tfidf":
			print("--TF IDF vectorizer")
			return TfidfVectorizer(ngram_range = self.ngram)
