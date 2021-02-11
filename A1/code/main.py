import pandas as pd
from sklearn import metrics

#Local Imports
from dataformat.data_class import Data
from dataformat.vectorizer_class import Vectorizer
from model.classifier_class import Classifier

# Import and Clean Raw Data
train_data = Data("data","train", reload = False)
test_data = Data("data","test", reload = False)

print("\ntraining data\n", train_data.processed_df['BLM'].value_counts())
print("\ntest data\n", test_data.processed_df['BLM'].value_counts())

#Vectorize Train and Test Data
# https://towardsdatascience.com/building-a-sentiment-classifier-using-scikit-learn-54c8e7c5d2f0
v = Vectorizer("tfidf", ngram=(1,1))

train_vocab = v.vectorizer.fit_transform(train_data.processed_df['text']).toarray()
train_X = pd.DataFrame(train_vocab, columns=v.vectorizer.get_feature_names())
train_Y = train_data.processed_df['BLM']

test_vocab = v.vectorizer.transform(test_data.processed_df['text']).toarray()
test_X = pd.DataFrame(test_vocab, columns=v.vectorizer.get_feature_names())
test_Y = test_data.processed_df['BLM']

print("number of features: ", len(train_X.columns))
print("number of training samples : ", len(train_Y))
print("number of test samples: ", len(test_Y))


#Choose Classifier
list_of_classifiers = [
    "multinomial_bayes",
    "gaussian_bayes",
    "bernoulli_bayes",
    "complement_bayes",
    "logistic",
]

for model in list_of_classifiers:
    print("\nRunning:", model)
    c = Classifier(model)
    c.classifier.fit(train_X ,train_Y)

    # https://towardsdatascience.com/sentiment-analysis-introduction-to-naive-bayes-algorithm-96831d77ac91
    predicted = c.classifier.predict(test_X)
    accuracy_score = metrics.accuracy_score(predicted,test_Y)
    print(str('{:04.2f}'.format(accuracy_score*100)+ '%'))
