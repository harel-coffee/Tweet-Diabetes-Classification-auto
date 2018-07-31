"""
Author: Adrian Ahne
Date: 27-06-2018

Classification model of personal vs institutional (advertising, health information, spam) tweets

- Get manually labeled tweets from csv *
- either train word embeddings (fastText) or load already trained word embeddings (fastText)
-

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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime
from gensim.models import FastText
import multiprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM, Dense, Flatten
from keras.models import Sequential


DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"



def tweet_vectorizer(tweet, model):
    """
        Gets FastText vector for each word in the tweet and calculates word embedding
        for the whole tweet by taking the mean of all word - vectors

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
    """
        Create Keras model

        Parameters (to use a grid search) for the LSTM:
        -----------------------------------------------------------------
        loss:       list of loss functions
        optimizer:  list of optimizers
        dropout:    list of dropout rates
        reccurent_dropout: list of reccurent_dropouts

        Return
        -----------------------------------------------------------------
        Model
    """

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


def preprocess_tweet(tweet):
    """
        Preprocess tweets in the same way the word embeddings (FastText) are trained
    """
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



# path to the FastText word embeddings
PATH_TRAINED_FASTTEXT = "Trained_FastText_2018-07-24_18-17-00.model"



if __name__ == '__main__':


    # add path to utils directory to system path
    path = 'D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\utils'
    if path not in sys.path:
        sys.path.insert(0, path)

    from sys_utils import *


    # load preprocessing library
    load_library('D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\preprocess')

    from sklearn_utils import *
    from mongoDB_utils import *

    from preprocess import Preprocess
    prep = Preprocess()


    # load csv in which we manually labelled the users
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\manually_labeled_users_instVSpers_MoreInstTweets_30072018.csv"
    tweets_csv = pd.read_csv(path_tweets, sep=";",
                             converters={"tweet_proc": lambda x: x.strip("[]").replace("'", "").split(", ")})

    # get matrix of metadata of the tweets that are added to the classification
    # as further features
    #meta_data = get_meta_data_features(tweets_csv, manually_labelled_tweets)

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
    V = np.array([tweet_vectorizer(preprocess_tweet(tweet), model_ft) for tweet in tweets_raw])


    # choose algo:
    #---------------------------------------------------------------------------

    #model = MultinomialNB()
    model = SVC()
    #model = LogisticRegression()
    #model = RandomForestClassifier()
    #model = XGBClassifier()
    #model = MLPClassifier(early_stopping=True, batch_size=32)


    #pipeline = create_pipeline(model, meta_data=[], user_description=desc.values)
    #pipeline = create_pipeline(model, meta_data=meta_data.values, user_description=desc.values)
    pipeline  = Pipeline([
                    ('model', model),
                ])


    # parameter grid for grid search by using fastText embeddings
    parameters_ft = {
                  # param for MultinomialNB
                  #'model__alpha': (10, 5, 1, 0.5, 0.1),

                  # param for LogisticRegression
                  #'model__C' : [10, 1.0, 0.1],
                  #'model__tol' : [1e-10, 1e-9],

                  # param for SVC
                  'model__kernel' : ["linear", "poly", "rbf"],
                  'model__C' : [15.0,12.0,10.0,],
                  'model__tol' : [1e-2, 1e-3],

                  # param for RandomForestClassifier
                  #'model__n_estimators' : [50, 60, 80],
                  #'model__criterion' : ['gini', 'entropy'],
                  #'model__max_features' : ['auto', 'log2'],
                  #'model__max_depth' : [40, 50]

                  # param for XGBoost Best: 0.910828 using {'model__learning_rate': 0.05, 'model__reg_alpha': 0, 'model__max_depth': 3, 'model__reg_lambda': 1.5, 'model__n_estimators': 300}
                  #'model__max_depth' : [3,4],
                  #'model__learning_rate' : [0.03, 0.05, 0.07],
                  #'model__booster' : ["gblinear"], #["gbtree", "gblinear", "dart"],
                  #'model__gamma' : [0, 0.01],
                  #'model__n_estimators' : [200, 300, 400, 500],
                  #'model__reg_alpha' : [0, 0.1],
                  #'model__reg_lambda' : [0.5, 1.0, 1.5]

                  # param for Multi layer perceptron
                  #'model__hidden_layer_sizes' : [(64), (64, 32), (32, 32)],
                  #'model__activation' : ['relu', 'tanh', 'logistic'],
                  #'model__solver' : ['adam', 'sgd'],
                  #'model__learning_rate' : ['constant', 'invscaling'],
                  #'model__tol' : [1e-3, 1e-4, 1e-5],

                  #'model__hidden_layer_sizes' :  [(64), (32), (16,16)],#[(64), (16, 16), (32, 16)],
                  #'model__activation' : ['relu'],# ['relu', 'tanh'],
                  #'model__solver' : ['adam'],#['adam', 'sgd'],
                  #'model__learning_rate' : ['constant', 'invscaling'],
                  #'model__tol' : [1e-2, 1e-3, 1e-4],
                  #'model__alpha' : [ 1e-4, 1e-5, 1e-6],
                  #'model__max_iter' : [200, 300],
                  #'model__beta_1' : [0.990, 0.999],
                  #'model__beta_2' : [1e-7, 1e-8, 1e-9]

    }


    # Two options:
    #   1) Grid search to find best model
    #   2) Train best model and save to disk

    # Option 1) Grid search to find best model
    # print("Start Grid search...")
    #grid = GridSearchCV(pipeline, parameters_ft, cv=10, n_jobs=-1, verbose=2)

    #grid = grid.fit(V, labels.values)

    #print(grid.cv_results_)
    #print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))


    # Option 2) Train best model and save to disk
    print("Train best model:")
    # train best model
    best_model = SVC()
    best_params = {
                  'model__tol': 1e-2,
                  'model__C' : 12.0,
                  'model__kernel':'rbf'
    }

    # create best pipeline
    best_pipeline = pipeline
    best_pipeline.set_params(**best_params)

    print("Train model...")
    best_pipeline_trained = best_pipeline.fit(V, labels.values)

    # save best model
    print("Save model to file...")
    file_name = "best_model_user_classif_{}.sav".format(datetime.datetime.now().strftime(DATE_FORMAT))
    joblib.dump(best_pipeline_trained, file_name)

    #import ipdb; ipdb.set_trace()


"""
LogisticRegression
With more institutional tweets to train
Best: 0.887033 using {'model__tol': 1e-10, 'model__C': 1.0}

SVC
old:
Best: 0.862378 using
{'model__kernel': 'rbf', 'model__C': 10.0, 'model__tol': 0.01}

With more insitutional tweets to train:
Best: 0.894066 using {'model__C': 12.0, 'model__tol': 0.01, 'model__kernel': 'rb
f'}



MultiLayerPerceptron
Best: 0.868078 using
{'model__learning_rate': 'constant', 'model__alpha': 1e-06,  'model__solver': 'adam',
 'model__tol': 0.0001, 'model__hidden_layer_sizes': (16, 16), 'model__activation': 'tanh'}

With more insitutional tweets to train:
Best: 0.895385 using
{'model__solver': 'adam', 'model__learning_rate': 'constant
', 'model__activation': 'relu', 'model__max_iter': 300, 'model__alpha': 1e-06, '
model__tol': 0.001, 'model__hidden_layer_sizes': 64}


RandomForest
Best: 0.860749 using
{'model__max_depth': 40, 'model__criterion': 'gini', 'model
__max_features': 'log2', 'model__n_estimators': 60}
"""
