"""
Author: Adrian Ahne
Date: 27-06-2018

Finding and training best model to classify personal users vs
institutional (advertising, health information, spam) users based
on their tweets


- Get manually labeled tweets from csv
- either train word embeddings (fastText) or load already trained word embeddings (fastText)
- Two options:
                1) Grid search to find best model
                2) Train best model and save to disk

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
import gensim
from gensim.models import FastText
#from gensim.models.wrappers import FastText
import multiprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM, Dense, Flatten
from keras.models import Sequential


# CONSTANTS
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

# path to the word embeddings
base = "D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\training\\trained_wordEmbeddings_10-08-2018\\"
ft_fullPrep = "FastText\\fullPreprocessing_09-08-2018\\ft_fullPrep_10-08-2018.model"
ft_partialPrep = "FastText\\partialPreprocessing_09-08-2018\\partialPreprocessing_09-08-2018.model"
ft_noPrep = "FastText\\noPreprocessing_09-08-2018\\noPreprocessing_09-08-2018.model"
ft_pretrained_wiki = "D:\\A_AHNE1\\Data\\FastText_Wikipedia_300\\wiki.en.bin"

# --------------------------------------------------------------------


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






if __name__ == '__main__':


    # Choose keyword of the 4 possible configurations
    # - "full_preprocessing" : use word embeddings with full preprocessing of the tweets
    # - "partial_preprocessing" : use word embeddings with partial preprocessing of the tweets
    # - "no_preprocessing" : use word embeddings with no tweet preprocessing
    # - "pretrained" : use word embeddings of downloaded pretrained corpus
    MODE = "no_preprocessing"

    if MODE == "full_preprocessing":
        PATH_TRAINED_FASTTEXT = base+ft_fullPrep
    elif MODE == "partial_preprocessing":
        PATH_TRAINED_FASTTEXT = base+ft_partialPrep
    elif MODE == "no_preprocessing":
        PATH_TRAINED_FASTTEXT = base+ft_noPrep
    elif MODE == "pretrained":
        PATH_TRAINED_FASTTEXT = ft_pretrained_wiki
    else:
        print("ERROR: Given mode {} not an option. Choose between ['full_preprocessing', 'partial_preprocessing', 'no_preprocessing', 'pretrained']".format(MODE))
        sys.exit(1)


    # add path to utils directory to system path
    path = 'D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\utils'
    if path not in sys.path:
        sys.path.insert(0, path)

    from tweet_utils import *


    # load csv in which we manually labelled the users
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\data\\manual_labeled_tweets\\manually_labeled_users_instVSpers_MoreInstTweets_30072018.csv"
    tweets_csv = pd.read_csv(path_tweets, sep=";",
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
        print("Load FastText model...")

        if MODE == "pretrained" :
            model_ft = gensim.models.wrappers.FastText.load_fasttext_format(PATH_TRAINED_FASTTEXT)
        else:
            model_ft = FastText.load(PATH_TRAINED_FASTTEXT)


    # Adapt preprocessing function depending on which preprocessed word embeddings you use!
    print("Get word embeddings for each tweet..")
    if MODE == "full_preprocessing":
        V = np.array([tweet_vectorizer(preprocess_tweet(tweet, mode="full_preprocessing"), model_ft) for tweet in tweets_raw])
    elif MODE == "partial_preprocessing" :
        V = np.array([tweet_vectorizer(preprocess_tweet(tweet, mode="partial_preprocessing"), model_ft) for tweet in tweets_raw])
    else:
        V = np.array([tweet_vectorizer(preprocess_tweet(tweet, mode="no_preprocessing"), model_ft) for tweet in tweets_raw])


    # remove the tweets that are empty because there is no word embedding
    VV = [(tweet, label) for tweet, label in zip(V, labels) if len(tweet) > 0]

    labels = pd.DataFrame([tup[1] for tup in VV])
    V = [tup[0] for tup in VV]


    # choose algo:
    #---------------------------------------------------------------------------

    #model = MultinomialNB(random_state=0)
    model = SVC(random_state=0)
    #model = LogisticRegression(random_state=0)
    #model = RandomForestClassifier(random_state=0)
    #model = XGBClassifier(random_state=0)
    #model = MLPClassifier(early_stopping=True, batch_size=32, random_state=0)


    pipeline  = Pipeline([
                    ('model', model),
                ])


    # parameter grid for grid search by using fastText embeddings
    parameters_ft = {
                  # param for MultinomialNB
                  #'model__alpha': (10, 5, 1, 0.5, 0.1),

                  # param for LogisticRegression
                  #'model__C' : [5, 3, 1.0, 0.8, 0.5, 0.1],
                  #'model__tol' : [1e-10, 1e-9],

                  # param for SVC
                  'model__kernel' : ["linear", "poly", "rbf"],
                  'model__C' : [ 12.0, 10.0, 8.0, 6.0, 4.0, 1.0],
                  'model__tol' : [1e-1, 1e-2, 1e-3],

                  # param for RandomForestClassifier
                  #'model__n_estimators' : [70, 80, 100, 120],
                  #'model__criterion' : ['gini', 'entropy'],
                  #'model__max_features' : ['auto', 'log2'],
                  #'model__max_depth' : [ 8, 10, 20]

                  # param for XGBoost Best: 0.910828 using {'model__learning_rate': 0.05, 'model__reg_alpha': 0, 'model__max_depth': 3, 'model__reg_lambda': 1.5, 'model__n_estimators': 300}
                  #'model__max_depth' : [3,4],
                  #'model__learning_rate' : [0.03, 0.05, 0.07],
                  #'model__booster' : ["gblinear"], #["gbtree", "gblinear", "dart"],
                  #'model__gamma' : [0, 0.01],
                  #'model__n_estimators' : [200, 300, 400, 500],
                  #'model__reg_alpha' : [0, 0.1],
                  #'model__reg_lambda' : [0.5, 1.0, 1.5]

                  # param for Multi layer perceptron
                  #'model__hidden_layer_sizes' : [(64,64), (64,32)],#[(64), (64, 32), (32, 32)],
                  #'model__activation' : ['relu', 'tanh', 'logistic'],
                  #'model__solver' : ['adam', 'sgd'],
                  #'model__learning_rate' : ['constant', 'invscaling'],
                  #'model__tol' : [1e-2, 1e-3],#[1e-2, 1e-3, 1e-4],

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
    """
    print("Start Grid search...")
    grid = GridSearchCV(pipeline, parameters_ft, cv=10, n_jobs=14, verbose=2)

    grid = grid.fit(V, labels.values)

    #print(grid.cv_results_)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))
    """

    # Option 2) Train best model and save to disk
    print("Train best model:")
    # train best model
    best_model = SVC()  # training acc of 90.20%; no preprocessing
    best_params = {
                  'model__tol': 0.01,
                  'model__C' : 6.0,
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



"""
NEW ####################################################
SVC full preprocessing embeddings
Best: 0.883516 using
{'model__C': 15.0, 'model__tol': 0.01, 'model__kernel': 'rbf'}


SVC partial preprocessing embeddings
Best: 0.896703 using
{'model__tol': 0.1, 'model__kernel': 'rbf', 'model__C': 12.0}


SVC no preprocessing embeddings
Best: 0.901978 using
{'model__C': 6.0, 'model__tol': 0.01, 'model__kernel': 'rbf'}


SVC pretrained Wikipedia corpus
Best: 0.865495 using
{'model__tol': 0.1, 'model__C': 10.0, 'model__kernel': 'rbf'}
------------------------------------------------------------------------

Logistic Regression full preprocessing embeddings
Best: 0.883956
using {'model__tol': 1e-10, 'model__C': 0.1}

Logistic Regression partial preprocessing embeddings
Best: 0.891868 using
{'model__tol': 1e-10, 'model__C': 0.8}

Logistic Regression no preprocessing embeddings
Best: 0.894505 using
{'model__tol': 1e-10, 'model__C': 0.8}

Logistic Regression pretrained Wikipedia corpus
Best: 0.870330 using {'model__C': 3, 'model__tol': 1e-10}
---------------------------------------------------------------------------

RandomForest full preprocessing
Best: 0.876484 using
{'model__criterion': 'entropy', 'model__max_features': 'auto',
 'model__max_depth': 40, 'model__n_estimators': 60}

RandomForest partial preprocessing
Best: 0.886154
{'model__n_estimators': 80, 'model__max_depth': 8, 'model__max_features': 'auto',
 'model__criterion': 'entropy'}

RandomForest noPreprocessing
Best: 0.884835 using
{'model__n_estimators': 120, 'model__criterion': 'gini', 'm
odel__max_features': 'log2', 'model__max_depth': 10}

RandomForest pretrained Wikipedia corpus
Best: 0.845714 using
{'model__max_features': 'auto', 'model__max_depth': 8, 'mod
el__criterion': 'entropy', 'model__n_estimators': 70}
--------------------------------------------------------


MultiLayerPerceptron no preprocessing
Best: 0.894505 using
 {'model__tol': 0.01, 'model__activation': 'relu', 'model__learning_rate': 'constant',
  'model__hidden_layer_sizes': (64, 64), 'model__solver': 'adam'}


"""





"""
OLD  ##############################################################


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
