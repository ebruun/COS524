from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

class Classifier():
    def __init__(self, which_classifier):
        #print("Made a classifier object...")

        self.name = which_classifier
        self.classifier = self.create_classifier()

    def create_classifier(self):

        if self.name == "multinomial_bayes":
            #print("Multinomial Bayes")
            return MultinomialNB()
        if self.name == "gaussian_bayes":
            #print("Gaussian Bayes")
            return GaussianNB()
        if self.name == "bernoulli_bayes":
            #print("Bernoulli Bayes")
            return BernoulliNB()
        if self.name == "complement_bayes":
            #print("Complement Bayes")
            return ComplementNB()
        elif self.name == "logistic":
            #print("Logistic")
            return LogisticRegression()
