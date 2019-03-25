"""
Author: Adrian Ahne
Date: 16-08-2018

Finding and training best model to classify personal tweets vs
institutional (advertising, health information, spam) tweets


- Get manually labeled tweets from csv
- Choose which word embedding to take
- Choose ML model (SVC, Logistic Regression, etc.)
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
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime
from gensim.models import FastText
import gensim
import multiprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM, Dense, Flatten
from keras.models import Sequential
import os



# CONSTANTS
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

# path to the word embeddings
base = "D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\training\\trained_wordEmbeddings_10-08-2018\\"
ft_fullPrep = "FastText\\fullPreprocessing_09-08-2018\\ft_fullPrep_10-08-2018.model"
ft_partialPrep = "FastText\\partialPreprocessing_09-08-2018\\partialPreprocessing_09-08-2018.model"
ft_noPrep = "FastText\\noPreprocessing_09-08-2018\\noPreprocessing_09-08-2018.model"
ft_pretrained_wiki = "D:\\A_AHNE1\\Data\\FastText_Wikipedia_300\\wiki.en.bin"






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
    model.compile(loss = loss, optimizer=optimizer, metrics = ['accuracy'])

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



    # load csv in which we manually labelled the tweets
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\data\\manual_labeled_tweets\\manually_labeled_tweets_instVSpers_25072018.csv"
    #path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\man_labeled_tweets_instVSpers_26072018_withoutUsersScoredLess02.csv"
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

    labels = pd.DataFrame([tup[1] for tup in VV]).values.ravel()
    V = np.array([tup[0] for tup in VV])


    # choose algo:
    #---------------------------------------------------------------------------

    #model = MultinomialNB()
    model = SVC(random_state=0)
    #model = LogisticRegression(random_state=0)
    #model = RandomForestClassifier(random_state=0)
    #model = XGBClassifier(random_state=0)
    #model = MLPClassifier(early_stopping=True, batch_size=32, random_state=0)
    #model = KerasClassifier(build_fn=create_model, epochs=10)

    #import ipdb; ipdb.set_trace()

    # !! Set when using Keras Classifier
    #V = np.reshape(V, (V.shape[0], V.shape[1], 1))
    #labels = np.reshape(labels, (labels[0], 1))



    pipeline  = Pipeline([
                    ('model', model),
                ])

    # parameter grid for grid search
    parameters = {
                  # param for MultinomialNB
                  #'model__alpha': (10, 5, 1, 0.5, 0.1),

                  # param for LogisticRegression
                  #'model__C' : [20.0, 10.0, 5.0, 1.0, 0.1],
                  #'model__tol' : [1e-10, 1e-9],

                  # param for SVC
                  #'model__kernel' : ["linear", "poly", "rbf"],
                  #'model__C' : [50.0, 45.0, 40.0, 35.0, 30.0],
                  #'model__tol' : [1e-2, 1e-3],

                  # param for RandomForestClassifier
                  #'model__n_estimators' : [30, 40, 50, 60],
                  #'model__criterion' : ['gini', 'entropy'],
                  #'model__max_features' : ['auto', 'log2', None],
                  #'model__max_depth' : [20, 30, 40]

                  # param for XGBoost Best: 0.910828 using {'model__learning_rate': 0.05, 'model__reg_alpha': 0, 'model__max_depth': 3, 'model__reg_lambda': 1.5, 'model__n_estimators': 300}
                  #'model__max_depth' : [3,4],
                  #'model__learning_rate' : [0.03, 0.05, 0.07],
                  #'model__booster' : ["gbtree"],#["gbtree", "gblinear", "dart"],
                  #'model__gamma' : [0, 0.01],
                  #'model__n_estimators' : [200, 300, 400],
                  #'model__reg_alpha' : [0, 0.1],
                  #'model__reg_lambda' : [0.5, 1.0, 1.5]

                  # param for Multi layer perceptron
                  #'model__hidden_layer_sizes' : [(32), (64), (64, 32), (32, 32)],
                  #'model__activation' : ['relu', 'tanh', 'logistic'],
                  #'model__solver' : ['adam', 'sgd'],
                  #'model__learning_rate' : ['constant', 'invscaling'],
                  #'model__tol' : [1e-2, 1e-3, 1e-4],
#                  'model__alpha' : [ 1e-3, 1e-4, 1e-5],
#                  'model__max_iter' : [200, 300],
#                  'model__beta_1' : [0.990, 0.999],
#                  'model__beta_2' : [1e-7, 1e-8, 1e-9]

                   # Keras Classifier
                   'model__epochs' : [10],#[10,20],
                   'model__loss' : ["binary_crossentropy", "sparse_categorical_crossentropy"],
                   'model__optimizer' : ["adam", "sgd"],
                   'model__dropout' : [0.1, 0.2],
                   'model__reccurent_dropout' : [0.1, 0.2]

    }

    # Two options:
    #   1) Grid search to find best model
    #   2) Train best model and save to disk
    """
    # Option 1) Grid search to find best model
    grid = GridSearchCV(pipeline, parameters, cv=5, n_jobs=13, verbose=2)
    grid = grid.fit(V, labels)

    #print(grid.cv_results_)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))
    """
    # Option 2) Train best model and save to disk

    # Best accuracy with 92.72 % SVC with no preprocessing
    best_params = {
                    "model__C" : 45.0,
                    "model__tol": 0.01,
                    "model__kernel" : "rbf"
    }
    #best_model = pipeline.fit_transform(tweets, labels, **best_params)

    #X_train, X_test, y_train, y_test = train_test_split(V, labels.values, test_size=0.33, random_state=0)

    best_pipeline = pipeline

    best_pipeline.set_params(**best_params)

    print("Train model...")
    best_pipeline_trained = best_pipeline.fit(V, labels)
    y_pred = best_pipeline_trained.predict(V)

    #tweets_csv["y_pred"] = y_pred

    #tweets_csv.to_csv("checkCheck.csv", sep=";")

    #import ipdb; ipdb.set_trace()

    # save best model
    print("Save model to file...")
    file_name = "best_model_tweets_classif_SVC_{}.sav".format(datetime.datetime.now().strftime(DATE_FORMAT))
    joblib.dump(best_pipeline_trained, file_name)


"""
SVC full preprocessing
Best: 0.907113 using {'model__tol': 0.01, 'model__C': 30.0, 'model__kernel': 'rbf'}

SVC partial preprocessing
Best: 0.918259 using {'model__kernel': 'rbf', 'model__tol': 0.01, 'model__C': 30.0}

SVC no preprocessing
Best: 0.927282 using {'model__C': 45.0, 'model__tol': 0.01, 'model__kernel': 'rbf'}

SVC pre-trained downloaded
Best: 0.881104 using {'model__tol': 0.01, 'model__C': 50.0, 'model__kernel': 'rbf'}
----------------------------------------------------------------------------------------

Logistic Regression full_preprocessing
Best: 0.904459 using {'model__C': 1.0, 'model__tol': 1e-10}

Logisitic Regression partial_preprocessing
Best: 0.915605 using {'model__C': 1.0, 'model__tol': 1e-10}

Logisitc Regression no_preprocessing
Best: 0.927813 using {'model__tol': 1e-10, 'model__C': 1.0}

Logistic Regression pre-trained
Best: 0.882696 using {'model__C': 5.0, 'model__tol': 1e-10}
------------------------------------------------------------------------------------

Random Forest full_preprocessing
Best: 0.899151 using {'model__n_estimators': 40, 'model__max_features': 'auto',
'model__max_depth': 30, 'model__criterion': 'entropy'}

Random Forest partial_preprocessing
Best: 0.900212 using {'model__n_estimators': 50, 'model__max_depth': 30, 'model_
_criterion': 'gini', 'model__max_features': 'auto'}

Random Forest no_preprocessing
Best: 0.909766 using {'model__max_features': 'auto', 'model__max_depth': 30, 'mo
del__criterion': 'gini', 'model__n_estimators': 50}

Random Forest pretrained
Best: 0.845011 using {'model__n_estimators': 60, 'model__max_features': 'auto',
'model__max_depth': 20, 'model__criterion': 'entropy'}
------------------------------------------------------------------------------------

Multilayer Perceptron full_preprocessing
Best: 0.906582 using {'model__learning_rate': 'constant', 'model__tol': 0.001, '
model__hidden_layer_sizes': 64, 'model__solver': 'adam', 'model__activation': 'r
elu'}

Multilayer Perceptron partial preprocessing
Best: 0.917197 using {'model__tol': 0.001, 'model__activation': 'tanh', 'model__
solver': 'adam', 'model__learning_rate': 'constant', 'model__hidden_layer_sizes'
: 64}

Multilayer Perceptron no preprocessing
Best: 0.925159 using {'model__activation': 'tanh', 'model__learning_rate': 'cons
tant', 'model__hidden_layer_sizes': 64, 'model__solver': 'adam', 'model__tol': 0
.001}
------------------------------------------------------------------------------------



"""
