"""
Author: Adrian Ahne
Date: 27-06-2018

Classification model of personal vs institutional tweets
"""

from pymongo import MongoClient
from pprint import pprint
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime
from gensim.models import FastText
import multiprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM, Dense, Flatten
from keras.models import Sequential

from utils import *

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"



def connect_to_database(host='localhost', port=27017):
    """
        Connect to MongoDB database
    """

    try:
        client = MongoClient(host, port)
    except ConnectionFailure as e:
        sys.stderr.write("Could not connect to MongoDB: %s" % e)
        sys.exit(1)

    return client


def load_preprocess_library(path='D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\preprocess'):
    """
        load library of the given path
    """

    if path not in sys.path:
        sys.path.insert(0, path)




def get_preprocessed_tweets(collection):
    """
        Preprocess raw tweets

        parameters
        -------------------------------------------------------------------
        collection:     MongoDB collection from which tweets are extracted

        Return
        -------------------------------------------------------------------
        list with preprocessed tweets

    """


    def preprocess_tweet(tweet):
    #    tweet = prep.replace_contractions(tweet)
    #    tweet = prep.replace_special_words(tweet)
        tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="replace",
                                               mode_Hashtag="replace")
        tweet = prep.tokenize(tweet)
        tweet = prep.remove_punctuation(tweet)

    #    tweet = prep.preprocess_emojis(tweet)
    #    tweet = prep.preprocess_emoticons(tweet)
        tweet = prep.remove_non_ascii(tweet)
        tweet = prep.to_lowercase(tweet)
        tweet = prep.replace_numbers(tweet)

    #    tweet = prep.remove_stopwords(tweet, include_personal_words=True, include_negations=False)
        tweet = prep.lemmatize_verbs(tweet)
    #    tweet = prep.stem_words(tweet)
        return tweet

    list_tweets = [preprocess_tweet(tweet["text"]) for tweet in collection.find()]

    return list_tweets



def tweet_vectorizer(tweet, model):
    """
        Gets FastText vector for each word and calculates word embedding for
        each tweet

        Parameters
        -------------------------------------------------------------------
        tweet:      list of preprocessed tweets
        model:      model with trained word embeddings

        Return
        ---------------------------------------------------------------------
        list containing word embeddings for each tweet
    """

    tweet_vec =[]
    numw = 0
    for w in tweet:
        try:
            if numw == 0:
                tweet_vec = model[w]
            else:
                tweet_vec = np.add(tweet_vec, model[w])
            numw+=1

        except:
            print("no embedding for {} !!!!!!!!!!!!".format(w))

    return np.asarray(tweet_vec) / numw


def create_model(loss, optimizer, dropout, reccurent_dropout):

    model = Sequential()
    #model.add(Embedding(2000, embed_dim,input_length = X.shape[1], dropout = 0.2))
    model.add(LSTM(128, input_shape=(200, 1), dropout = dropout,
                   recurrent_dropout = reccurent_dropout, return_sequences=True))
#    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = loss, optimizer=optimizer,
                  metrics = ['accuracy'])

    return model




PATH_TRAINED_FASTTEXT = "Trained_FastText_2018-07-24_18-17-00.model"


if __name__ == '__main__':


    client = connect_to_database()

    # load preprocessing library
    load_preprocess_library()

    from preprocess import Preprocess
    prep = Preprocess()

    # get database with all tweets
    db = client.tweets_database

    # get collections
    english_noRetweet_tweets = db.english_noRetweet_tweets
    tweets_manual_label = db.tweets_manual_label

    # load csv in which we manually labelled the tweets
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\manually_labeled_tweets_instVSpers_17072018.csv"
    tweets_csv = pd.read_csv(path_tweets, sep=",",
                             converters={"tweet_proc": lambda x: x.strip("[]").replace("'", "").split(", ")})


    # get labels and tweet matrix
    labels = tweets_csv["personal (0=no, 1=yes)"]
    tweets_proc = tweets_csv["tweet_proc"]
    tweets_raw = tweets_csv["tweet"]


    # Train new FastText model
    if PATH_TRAINED_FASTTEXT == []:

        # list with preprocessed tweets
        #list_tweets = tweets_proc
        print("Preprocess tweets...")
        list_tweets = get_preprocessed_tweets(english_noRetweet_tweets)

        # word embedding Fasttext
        print("Train Fasttext...")
        model_ft = FastText(list_tweets, size=200, window=5, min_count=1,
                            workers=multiprocessing.cpu_count() ,sg=1, hs=0, iter=20,
                            word_ngrams=1, min_n=3, max_n=6)

        print("Save model to disk...")
        file_name = "Trained_FastText_{}.model".format(datetime.datetime.now().strftime(DATE_FORMAT))
        model_ft.save(file_name)

    # load trained model
    else:
        model_ft = FastText.load(PATH_TRAINED_FASTTEXT)


    print("Get word embeddings for each tweet..")
    V = np.array([tweet_vectorizer(tweet, model_ft) for tweet in tweets_proc])

    #V = np.reshape(V, V.shape + (1,))
    V = np.reshape(V, (V.shape[0], V.shape[1], 1))
    print(V.shape)
    # choose algo
    #model = MultinomialNB()
    #model = SVC()
    #model = RandomForestClassifier()
    #model = XGBClassifier()

    model = Sequential()
    #model.add(Embedding(2000, embed_dim,input_length = X.shape[1], dropout = 0.2))
    model.add(LSTM(128, input_shape=(200, 1), dropout = 0.2,
                   recurrent_dropout = 0.2, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',
                  metrics = ['accuracy'])
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(V, labels.values, test_size=0.33, random_state=0)

    hist = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=2)
    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

"""
    model = KerasClassifier(build_fn=create_model, epochs=10)

    param_grid = dict(epochs=[10], #[10,20],
                      loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
                      optimizer=["adam"],#["adam", "sgd"],
                      dropout=[0.1],#[0.1, 0.2],
                      reccurent_dropout = [0.1, 0.2])


    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

"""
