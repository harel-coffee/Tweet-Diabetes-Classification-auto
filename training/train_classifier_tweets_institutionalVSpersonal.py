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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime

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


def get_meta_data_features(tweets_csv, manually_labelled_tweets):
    """
        Get some meta-data of the labeled tweets

        Parameter:
          - tweets_csv : DataFrame with labeled tweets
          - manually_labelled_tweets : Collection in which all raw tweet information
                                       of the labeled tweets are stored
    """

    # define DataFrame
    meta_data_pd = pd.DataFrame(columns=["n_hashtags", "n_urls", "n_user_mentions",
                                         "followers_count", "friends_count"])

    for i, user_name in enumerate(tweets_csv["user_name"]):
        for user in manually_labelled_tweets.find({'user.screen_name' : user_name}):
            meta_data_pd.loc[i] = [len(user["entities"]['hashtags']),
                                   len(user["entities"]['urls']),
                                   len(user["entities"]['user_mentions']),
                                   user["user"]['followers_count'],
                                   user["user"]['friends_count']]

    return meta_data_pd


def create_pipeline(model, meta_data=[]):
    """
        Create Pipeline

        Parameters
        -------------------------------------------------------
        - model : algorithm for classification to be used
        - meta_data : meta_data information like number of followers / friends etc.

        Return
        --------------------------------------------------------
        pipeline object
    """

    # meta data given
    if meta_data != []:
        print("Create pipeline using meta-data...")
        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(meta_data=meta_data)),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(lowercase=False))
                    ])),

                    # Pipeline handling meta data
                    ('metadata', Pipeline([
                        ('metadataSelector', ItemSelector(key='metadata')),
                        ('tosparse', ArrayCaster()),
                        ('scale', StandardScaler(with_mean=False)),
                        ('selectKbest', SelectKBest(f_classif)),
                    ]))
                ]
            )),

            ('model', model),
        ])

    # no meta data and no user description
    else:
        print("Create pipeline...")

        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            #('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor()),


            ('tfidfvect', TfidfVectorizer(lowercase=False)),

            ('model', model),
        ])

    return pipeline


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
    tweets_csv = pd.read_csv(path_tweets, sep=",")

    # get matrix of metadata of the tweets that are added to the classification
    # as further features
    meta_data = get_meta_data_features(tweets_csv, tweets_manual_label)

    # get labels and tweet matrix
    labels = tweets_csv["personal (0=no, 1=yes)"]
    tweets = tweets_csv["tweet_proc"]

    # choose algo
    #model = MultinomialNB()
    #model = SVC()
    #model = RandomForestClassifier()
    model = XGBClassifier()


    pipeline = create_pipeline(model, meta_data=[])
    #pipeline = create_pipeline(model, meta_data=meta_data.values, user_description=desc.values)

    # parameter grid for grid search
    parameters = {'tfidfvect__ngram_range': [(1, 1), (1, 2)],
                  'tfidfvect__analyzer' : ["word"],#['word', 'char'],
                  'tfidfvect__min_df' : [1, 5],
                  'tfidfvect__max_df' : [0.9],#[0.9, 1.0],
                  'tfidfvect__use_idf': [True],#(True, False),
                  'tfidfvect__smooth_idf': [False],#(True, False),
                  'tfidfvect__sublinear_tf': [True],#(True, False),

                  # param for MultinomialNB
                  #'model__alpha': (5, 1, 0.5 ),

                  # param for LogisticRegression
                  #'model__C' : [35.0, 30.0, 25.0],
                  #'model__tol' : [1e-10, 1e-9],

                  # param for SVC
                  #'model__kernel' : ["linear", "poly", "rbf"],
                  #'model__C' : [5.0, 1.0, 0.1],
                  #'model__tol' : [1e-4, 1e-5],

                  # param for RandomForestClassifier
                  #'model__n_estimators' : [20, 30, 40],
                  #'model__criterion' : ['gini', 'entropy'],
                  #'model__max_features' : ['auto', 'log2', None],
                  #'model__max_depth' : [10, 20, 30]

                  # param for XGBoost
                  'model__max_depth' : [3,4],
                  'model__learning_rate' : [0.05, 0.1],
                  'model__booster' : ["gbtree", "gblinear", "dart"],
                  'model__gamma' : [0],#[0, 0.01],
                  'model__n_estimators' : [100, 200],
                  'model__reg_alpha' : [0],#[0, 0.1],
                  'model__reg_lambda' : [0.0, 0.1, 0.5], #[0.5, 1.0]

    }



    # GridSearchCV gets stuck for n_jobs != 1 when executed in jupyter notebook.
    # https://github.com/scikit-learn/scikit-learn/issues/5115
    # To fix bug set environment variable: export JOBLIB_START_METHOD="forkserver"
    #%env JOBLIB_START_METHOD="forkserver"
    grid = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=2)
    grid = grid.fit(tweets, labels)
    #pp = pipeline.fit_transform(tweets, labels)

    #print(grid.cv_results_)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))
"""
    # train best model
    best_model = MultinomialNB()
    best_params = {'union__tweets__tfidfvect__ngram_range': (1, 1),
                  'union__tweets__tfidfvect__analyzer' : 'word',
                  'union__tweets__tfidfvect__min_df' : 1,
                  'union__tweets__tfidfvect__max_df' : 0.9,
                  'union__tweets__tfidfvect__use_idf': False,
                  'union__tweets__tfidfvect__smooth_idf': True,
                  'union__tweets__tfidfvect__sublinear_tf': True,

                  'model__alpha': 1
    }

    # create best pipeline
    best_pipeline = create_pipeline(best_model,  meta_data=[], user_description=[])
    best_pipeline.set_params(**best_params)

    print("Train model...")
    best_pipeline_trained = best_pipeline.fit(tweets, labels)

    # save best model
    print("Save model to file...")
    file_name = "best_model_{}.sav".format(datetime.datetime.now().strftime(DATE_FORMAT))
    joblib.dump(best_pipeline_trained, file_name)

    #import ipdb; ipdb.set_trace()
"""

"""
MultinomialNB
Best: 0.873142 using
{'tfidfvect__ngram_range': (1, 2), 'tfidfvect__smooth_idf': True, 'model__alpha': 1,
 'tfidfvect__sublinear_tf': True, 'tfidfvect__use_idf': False, 'tfidfvect__min_df': 5,
 'tfidfvect__analyzer': 'word', 'tfidfvect__max_df': 0.9}

SVC
Best: 0.883758 using
{'tfidfvect__analyzer': 'word', 'tfidfvect__max_df': 0.9, 'tfidfvect__min_df': 1,
 'tfidfvect__ngram_range': (1, 2), 'model__kernel': 'linear',
 'tfidfvect__use_idf': True, 'model__C': 1.0, 'model__tol': 1e-05,
 'tfidfvect__smooth_idf': False, 'tfidfvect__sublinear_tf': False}

XGBoost
Best: 0.884289 using
{'model__reg_alpha': 0, 'model__learning_rate': 0.05, 'tfidfvect__use_idf': True,
 'model__reg_lambda': 0.5, 'tfidfvect__max_df': 0.9, 'model__max_depth': 3,
 'model__booster': 'gblinear', 'tfidfvect__min_df': 1, 'model__gamma': 0,
 'tfidfvect__sublinear_tf': True, 'tfidfvect__ngram_range': (1, 2),
 'tfidfvect__smooth_idf': False, 'tfidfvect__analyzer': 'word', 'model__n_estimatorqs': 200}
"""
