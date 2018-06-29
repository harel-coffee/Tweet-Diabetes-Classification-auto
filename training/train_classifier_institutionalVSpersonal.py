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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

from utils import *



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


def create_pipeline(model, meta_data_pd, add_metadata=False):
    """
        Create Pipeline

        Parameters
        -------------------------------------------------------
        - model : algorithm for classification to be used
        - add_metadata : (True, False)
                         if True, add meta_data like number of followers / friends etc.

        Return
        --------------------------------------------------------
        pipeline object
    """

    if add_metadata:
        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(meta_data_pd.values,
                                                                             add_metadata=True)),

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

    # without metadata
    else:
        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(meta_data_pd.values,
                                                                             add_metadata=False)),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(lowercase=False)),
                    ]))
                ]
            )),

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
    manually_labelled_tweets = db.manually_labelled_tweets

    # load csv in which we manually labelled the tweets
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\manually_labeled_tweets_users_instVSpers_noNeg.csv"
    tweets_csv = pd.read_csv(path_tweets, sep=";")

    # get matrix of metadata of the tweets that are added to the classification
    # as further features
    meta_data = get_meta_data_features(tweets_csv, manually_labelled_tweets)
#    import ipdb; ipdb.set_trace()

    # get labels and tweet matrix
    labels = tweets_csv["personal (0=no, 1=yes)"]
    tweets = tweets_csv["tweet_proc"]

    # choose algo
    #model = MultinomialNB()
    #model = SVC()
    model = RandomForestClassifier()


    pipeline = create_pipeline(model, meta_data, add_metadata=False)

    # parameter grid for grid search
    parameters = {#'union__tweets__tfidfvect__ngram_range': [(1, 1), (1, 2)],
                  'union__tweets__tfidfvect__min_df' : [1, 5],
                  'union__tweets__tfidfvect__max_df' : [0.9],
                  'union__tweets__tfidfvect__use_idf': (True, False),
                  'union__tweets__tfidfvect__smooth_idf': (True, False),
                  'union__tweets__tfidfvect__sublinear_tf': (True, False),

                  #'union__metadata__scale__with_std': (True, False),
                  #'union__metadata__selectKbest__k': [0, 1],

                  # param for MultinomialNB
                  #'model__alpha': (10, 5, 1, 1e-1),

                  # param for LogisticRegression
                  #'model__C' : [ 15.0, 20.0, 25.0],
                  #'model__tol' : [1e-7, 1e-6],

                  # param for SVC
                  #'model__kernel' : ["linear", "poly", "rbf"],
                  #'model__C' : [0.1, 1.0, 10],
                  #'model__tol' : [1e-6, 1e-5],

                  # param for RandomForestClassifier
                  'model__n_estimators' : [20, 30, 40],
                  'model__criterion' : ['gini', 'entropy'],
                  'model__max_features' : ['auto', 'log2', None],
                  'model__max_depth' : [10, 20, 30]
    }

    # GridSearchCV gets stuck for n_jobs != 1 when executed in jupyter notebook.
    # https://github.com/scikit-learn/scikit-learn/issues/5115
    # To fix bug set environment variable: export JOBLIB_START_METHOD="forkserver"
    #%env JOBLIB_START_METHOD="forkserver"
    grid = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=2)
    grid = grid.fit(tweets, labels)

    print(grid.cv_results_)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))


"""
MultinomialNB
Best: 0.846271 using
{'union__metadata__scale__with_std': True, 'union__tweets__tfidfvect__use_idf': False,
 'union__tweets__tfidfvect__smooth_idf': True, 'union__metadata__selectKbest__k': 0,
 'union__tweets__tfidfvect__min_df': 1, 'union__tweets__tfidfvect__max_df': 0.9,
 'model__alpha': 1, 'union__tweets__tfidfvect__sublinear_tf': True}

MultinomialNB without metadata
Best: 0.846271 using
{'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__smooth_idf': True,
 'model__alpha': 1, 'union__tweets__tfidfvect__min_df': 1,
 'union__tweets__tfidfvect__sublinear_tf': True, 'union__tweets__tfidfvect__use_idf': False}

SVC
Best: 0.832572 using
{'union__tweets__tfidfvect__min_df': 1, 'model__kernel': 'linear',
 'union__tweets__tfidfvect__sublinear_tf': True,
 'union__tweets__tfidfvect__smooth_idf': True, 'model__tol': 1e-06, 'model__C': 1.0,
 'union__tweets__tfidfvect__use_idf': False, 'union__metadata__selectKbest__k': 0,
 'union__tweets__tfidfvect__max_df': 0.9}

LogisticRegression
Best: 0.829528 using
{'union__tweets__tfidfvect__use_idf': True, 'model__C': 25.0,
 'union__tweets__tfidfvect__min_df': 5, 'union__tweets__tfidfvect__sublinear_tf': True,
 'union__tweets__tfidfvect__smooth_idf': True, 'union__metadata__selectKbest__k': 0,
 'model__tol': 1e-07, 'union__tweets__tfidfvect__max_df': 0.9}

RandomForest
Best: 0.832572 using
{'union__tweets__tfidfvect__sublinear_tf': False, 'model__criterion': 'gini',
 'model__max_depth': 30, 'union__tweets__tfidfvect__use_idf': True,
 'model__n_estimators': 40, 'union__tweets__tfidfvect__max_df': 0.9,
 'union__tweets__tfidfvect__smooth_idf': False, 'union__metadata__selectKbest__k': 0,
 'union__tweets__tfidfvect__min_df': 5, 'model__max_features': 'log2'}

RandomForest without metadata  
Best: 0.832572 using
{'model__n_estimators': 40, 'model__max_depth': 30, 'model__criterion': 'gini',
 'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__sublinear_tf': False,
 'model__max_features': 'log2', 'union__tweets__tfidfvect__min_df': 5,
 'union__tweets__tfidfvect__smooth_idf': False, 'union__tweets__tfidfvect__use_idf': False}

"""
