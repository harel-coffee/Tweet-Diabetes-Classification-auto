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
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime
from gensim.models import FastText
import multiprocessing

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"



if __name__ == '__main__':


    # add path to utils directory to system path
    path = 'D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\utils'
    if path not in sys.path:
        sys.path.insert(0, path)

    from sys_utils import *
    from tweet_utils import *
    from mongoDB_utils import *


    client = connect_to_database()

    # get database with all tweets
    db = client.tweets_database

    # get collections
    english_noRetweet_tweets = db.english_noRetweet_tweets
    users_manual_label_all_tweets = db.users_manual_label_all_tweets


    # load csv in which we manually labelled the users
    path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\manually_labeled_users_instVSpers_MoreInstTweets_30072018_withDescription.csv"
    tweets_csv = pd.read_csv(path_tweets, sep=";",
                             converters={"tweet_proc": lambda x: x.strip("[]").replace("'", "").split(", "),
                                         "desc_proc": lambda x: x.strip("[]").replace("'", "").split(", ")})


    # get matrix of metadata of the tweets that are added to the classification
    # as further features
    #meta_data = get_meta_data_features(tweets_csv, users_manual_label_all_tweets)

    # get labels and tweet matrix
    labels = tweets_csv["personal (0=no, 1=yes)"]
    tweets = tweets_csv["tweet_proc"]
    desc = tweets_csv["desc_proc"]

    # choose algo
    #model = MultinomialNB(random_state=0)
    #model = SVC(random_state=0)
    #model = RandomForestClassifier(random_state=0)
    #model = XGBClassifier(random_state=0)
    model = LogisticRegression(random_state=0)


    #pipeline = create_pipeline_BoW(model, meta_data=[], user_description=desc.values)
    pipeline = create_pipeline_BoW(model, meta_data=[], user_description=[])
    #pipeline = create_pipeline_BoW(model, meta_data=meta_data.values, user_description=desc.values)

    # parameter grid for grid search
    parameters = {'union__tweets__tfidfvect__ngram_range': [(1, 1), (1, 2)],
                  'union__tweets__tfidfvect__analyzer' : ['word'],#['word', 'char'],
                  'union__tweets__tfidfvect__min_df' : [1, 5],
                  'union__tweets__tfidfvect__max_df' : [0.9],#[0.9, 1.0],
                  'union__tweets__tfidfvect__use_idf': (True, False),
                  'union__tweets__tfidfvect__smooth_idf': (True, False),
                  'union__tweets__tfidfvect__sublinear_tf': (True, False),

                  # 'union__desc__tfidfvect__ngram_range': [(1, 1), (1, 2)],
                  # 'union__desc__tfidfvect__analyzer' : ["word"],#['word', 'char'],
                  # 'union__desc__tfidfvect__min_df' : [1, 5],
                  # 'union__desc__tfidfvect__max_df' : [0.9],#[0.9, 1.0],
                  # 'union__desc__tfidfvect__use_idf': (True, False),
                  # 'union__desc__tfidfvect__smooth_idf': (True, False),
                  # 'union__desc__tfidfvect__sublinear_tf': (True, False),

#                  'union__metadata__scale__with_std': (True, False),
#                  'union__metadata__selectKbest__k': [0, 2, 'all'],

#                  'union__transformer_weights': [{"tweets":1, "desc":1},
#                                                 {"tweets":1, "desc":0.5},
                                                 #{"tweets":1, "desc":0.2},
#                                                 {"tweets":1, "desc":0.0},
                                                 #{"tweets":1, "desc":0.0}
#                                                 ],

                  # param for MultinomialNB
                  #'model__alpha': ( 1, 0.1, 0.1),

                  # param for LogisticRegression
                  'model__C' : [40.0, 35.0],
                  'model__tol' : [1e-10, 1e-9],

                  # param for SVC
                  #'model__kernel' : ["linear", "poly", "rbf"],
                  #'model__C' :[35.0, 30.0, 25.0],
                  #'model__tol' : [1e-10, 1e-9],

                  # param for RandomForestClassifier
                  #'model__n_estimators' : [ 30, 40],
                  #'model__criterion' : ['gini', 'entropy'],
                  #'model__max_features' : ['auto', 'log2', None],
                  #'model__max_depth' : [10, 20, 30]

                  # param for XGBoost
                  #'model__max_depth' : [3,4],
                  #'model__learning_rate' : [0.01, 0.05, 0.7],
                  #'model__booster' : ["gblinear"], #["gbtree", "gblinear", "dart"],
                  #'model__gamma' : [0, 0.01],
                  #'model__n_estimators' : [100, 200, 300],
                  #'model__reg_alpha' : [0.0], #[0, 0.1],
                  #'model__reg_lambda' : [1.0]#[0.5, 1.0, 1.5]

    }


    # GridSearchCV gets stuck for n_jobs != 1 when executed in jupyter notebook.
    # https://github.com/scikit-learn/scikit-learn/issues/5115
    # To fix bug set environment variable: export JOBLIB_START_METHOD="forkserver"
    #%env JOBLIB_START_METHOD="forkserver"
    grid = GridSearchCV(pipeline, parameters, cv=10, n_jobs=14, verbose=2)
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
    best_pipeline = create_pipeline_BoW(best_model,  meta_data=[], user_description=[])
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
NEW =====================================================
WITH MORE SAMPLES

---------------------
MultinomialNB
Best: 0.877802 using {'model__alpha': 0.1, 'union__tweets__tfidfvect__max_df':.9,
'union__tweets__tfidfvect__smooth_idf': True, 'union__tweets__tfidfvect__u_idf': False,
'union__tweets__tfidfvect__analyzer': 'word', 'union__tweets__tffvect__min_df': 5,
'union__tweets__tfidfvect__ngram_range': (1, 2), 'union__twts__tfidfvect__sublinear_tf': False}

Multinomial with metadata
Best: 0.877802 using {'union__tweets__tfidfvect__ngram_range': (1, 2),
 'union__tweets__tfidfvect__use_idf': False, 'model__alpha': 0.1,
 'union__tweets__tfidfvect__min_df': 5, 'union__metadata__selectKbest__k': 0,
 'union__tweets__tfidfvect__analyzer': 'word', 'union__tweets__tfidfvect__smooth_idf': True,
 'union__tweets__tfidfvect__max_df': 0.9, 'union__metadata__scale__with_std': True,
 'union__tweets__tfidfvect__sublinear_tf': False}

Multinomial with description
Best: 0.785055 using {'union__tweets__tfidfvect__use_idf': False,
'union__desc__tfidfvect__min_df': 1, 'union__desc__tfidfvect__use_idf': True,
'union__desc__tfidfvect__ngram_range': (1, 1),
'model__alpha': 1, 'union__tweets__tfidfvect__sublinear_tf': False,
'union__tweets__tfidfvect__ngram_range': (1, 2), 'union__tweets__tfidfvect__min_df': 5,
'union__tweets__tfidfvect__smooth_idf': True, 'union__desc__tfidfvect__analyzer': 'word',
'union__desc__tfidfvect__sublinear_tf': True, 'union__desc__tfidfvect__smooth_idf': True,
'union__desc__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__max_df': 0.9,
'union__tweets__tfidfvect__analyzer': 'word'}
--------------------------


SVC
Best: 0.880000 using {'model__tol': 1e-10, 'model__C': 30.0,
'union__tweets__tfidfvect__min_df': 1, 'union__tweets__tfidfvect__ngram_range': (1, 2),
'union__tweets__tfidfvect__sublinear_tf': True, 'union__tweets__tfidfvect__max_df': 0.9,
'union__tweets__tfidfvect__use_idf': True, 'model__kernel': 'linear',
'union__tweets__tfidfvect__smooth_idf': False, 'union__tweets__tfidfvect__analyzer': 'word'}


SVC with metadata
Best: 0.880000 using
{'model__kernel': 'linear', 'union__tweets__tfidfvect__use_idf': True, 'union__tweets__tfidfvect__min_df': 1, 'model__C': 30.0, 'union__twe
ets__tfidfvect__sublinear_tf': True, 'union__tweets__tfidfvect__ngram_range': (1
, 2), 'model__tol': 1e-10, 'union__metadata__selectKbest__k': 0, 'union__tweets_
_tfidfvect__smooth_idf': False, 'union__tweets__tfidfvect__max_df': 0.9, 'union_
_tweets__tfidfvect__analyzer': 'word'}

SVC with description
Best: 0.825495 using
{'union__tweets__tfidfvect__use_idf': True, 'model__C': 35.0,
 'union__tweets__tfidfvect__analyzer': 'word', 'union__desc__tfidfvect__ngram_range': (1, 1),
 'union__tweets__tfidfvect__max_df': 0.9, 'union__desc__tfidfvect__max_df': 0.9,
 'union__tweets__tfidfvect__ngram_range': (1, 2), 'union__desc__tfidfvect__sublinear_tf': False,
 'union__tweets__tfidfvect__smooth_idf': False, 'model__kernel': 'linear',
 'union__desc__tfidfvect__min_df': 5, 'union__tweets__tfidfvect__min_df': 1,
 'union__desc__tfidfvect__analyzer': 'word', 'model__tol':1e-10,
 'union__desc__tfidfvect__use_idf': False, 'union__tweets__tfidfvect__sublinear_tf': True,
 'union__desc__tfidfvect__smooth_idf': True}
--------------------------------------------------------------

LogisticRegression
Best: 0.883077 using
{'union__tweets__tfidfvect__sublinear_tf': True, 'model__tol': 1e-10,
 'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__min_df': 1,
 'union__tweets__tfidfvect__use_idf': True, 'union__tweets__tfidfvect__analyzer': 'word',
 'union__tweets__tfidfvect__smooth_idf': False, 'model__C': 40.0,
 'union__tweets__tfidfvect__ngram_range': (1, 2)}

LogisticRegression with meta data
Best: 0.883516 using
{'model__C': 42.0, 'union__tweets__tfidfvect__use_idf': True,
 'union__tweets__tfidfvect__ngram_range': (1, 2), 'union__tweets__tfidfvect__sublinear_tf': True,
 'model__tol': 1e-10, 'union__tweets__tfidfvect__min_df': 1,
 'union__metadata__selectKbest__k': 0, 'union__tweets__tfidfvect__max_df': 0.9,
 'union__tweets__tfidfvect__analyzer': 'word', 'union__tweets__tfidfvect__smooth_idf': True}


LogisticRegression with user description
Best: 0.835604 using
{'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__ngram_range': (1, 2),
 'union__desc__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__min_df': 1,
 'union__desc__tfidfvect__smooth_idf': False, 'model__C': 40.0,
 'union__desc__tfidfvect__min_df': 1, 'model__tol': 1e-10,
 'union__desc__tfidfvect__sublinear_tf': True, 'union__desc__tfidfvect__analyzer': 'word',
 'union__desc__tfidfvect__use_idf': True, 'union__tweets__tfidfvect__smooth_idf': True,
 'union__tweets__tfidfvect__sublinear_tf': True, 'union__tweets__tfidfvect__analyzer': 'word',
 'union__tweets__tfidfvect__use_idf': True, 'union__desc__tfidfvect__ngram_range': (1, 2)}


-----------------------------------------------------------

RandomForest
Best: 0.872088 using
{'model__n_estimators': 40, 'union__tweets__tfidfvect__min_df': 5,
 'union__tweets__tfidfvect__smooth_idf': False, 'union__tweets__tfidfvect__use_idf': True,
 'union__tweets__tfidfvect__ngram_range': (1, 1), 'union__tweets__tfidfvect__sublinear_tf': True,
 'model__criterion': 'entropy', 'model__max_depth': 30, 'model__max_features': 'auto',
 'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__analyzer': 'word'}

RandomForest with meta_data
Best: 0.872088 using
{'model__criterion': 'entropy', 'union__tweets__tfidfvect__smooth_idf': False,
 'model__max_depth': 30, 'union__tweets__tfidfvect__min_df':5,
 'union__metadata__selectKbest__k': 0, 'union__tweets__tfidfvect__use_idf': True,
 'union__tweets__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__sublinear_tf': True,
 'union__tweets__tfidfvect__ngram_range': (1, 1), 'model__n_estimators': 40,
 'model__max_features': 'auto', 'union__tweets__tfidfvect__analyzer': 'word'}

Random Forest with users description
Best: 0.860659 using
{'union__tweets__tfidfvect__ngram_range': (1, 1), 'model__max_depth': 20,
'union__tweets__tfidfvect__sublinear_tf': False, 'union__tweets__tfidfvect__smooth_idf': False,
'union__desc__tfidfvect__sublinear_tf': False, 'union__tweets__tfidfvect__analyzer': 'word',
'union__desc__tfidfvect__analyzer':'word', 'union__desc__tfidfvect__use_idf': False,
'union__desc__tfidfvect__min_df': 1, 'union__desc__tfidfvect__max_df': 0.9,
'model__criterion': 'gini', 'model__n_estimators': 40,
'union__desc__tfidfvect__ngram_range': (1, 2), 'model__max_features': None,
'union__desc__tfidfvect__smooth_idf': True, 'union__tweets__tfidfvect__min_df': 1,
'union__tweets__tfidfvect__use_idf': True, 'union__tweets__tfidfvect__max_df': 0.9}


RandomForest with users description and weights for both matrices (but does not change anythin)
Best: 0.865934 using {'model__n_estimators': 40, 'model__max_depth': 20, 'union_
_desc__tfidfvect__min_df': 1, 'union__desc__tfidfvect__max_df': 0.9, 'union__twe
ets__tfidfvect__max_df': 0.9, 'union__transformer_weights': {'desc': 0.0, 'tweet
s': 1}, 'union__tweets__tfidfvect__use_idf': True, 'union__desc__tfidfvect__smoo
th_idf': True, 'union__tweets__tfidfvect__sublinear_tf': False, 'union__desc__tf
idfvect__sublinear_tf': False, 'model__max_features': None, 'union__tweets__tfid
fvect__smooth_idf': False, 'union__tweets__tfidfvect__analyzer': 'word', 'union_
_tweets__tfidfvect__ngram_range': (1, 1), 'union__desc__tfidfvect__use_idf': Fal
se, 'union__desc__tfidfvect__analyzer': 'word', 'model__criterion': 'gini', 'uni
on__tweets__tfidfvect__min_df': 1, 'union__desc__tfidfvect__ngram_range': (1, 1)
}


"""



"""
OLD ==============================
WITH LESS SAMPLES


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


Multinomial without metadata but with description
Best: 0.846906 using
{'union__transformer_weights': {'tweets': 1, 'desc': 0.2},
'union__tweets__tfidfvect__analyzer': 'word',
'union__tweets__tfidfvect__sublinear_tf': True, 'union__desc__tfidfvect__ngram_range': (1, 2),
'union__desc__tfidfvect__min_df': 1, 'union__tweets__tfidfvect__ngram_range': (1, 2),
'union__tweets__tfidfvect__use_idf': False, 'model__alpha': 1, 'union__desc__tfidfvect__use_idf': False,
'union__tweets__tfidfvect__min_df': 1, 'union__desc__tfidfvect__sublinear_tf': False,
'union__desc__tfidfvect__analyzer': 'word', 'union__tweets__tfidfvect__smooth_idf': True,
'union__desc__tfidfvect__smooth_idf': True, 'union__tweets__tfidfvect__max_df': 0.9,
'union__desc__tfidfvect__max_df': 0.9}

Multinomial with metadata and with user description
Best: 0.846091 using
{'union__desc__tfidfvect__ngram_range': (1, 2), 'union__tweets__tfidfvect__max_df': 0.9,
 'union__desc__tfidfvect__use_idf': False, 'union__tweets__tfidfvect__analyzer': 'word',
 'union__tweets__tfidfvect__sublinear_tf':True, 'union__desc__tfidfvect__max_df': 0.9,
 'union__tweets__tfidfvect__min_df': 1, 'union__transformer_weights': {'tweets': 1, 'desc': 0.2},
 'union__desc__tfidfvect__sublinear_tf': True, 'union__tweets__tfidfvect__use_idf': False,
 'union__desc__tfidfvect__min_df': 1, 'model__alpha': 1,
 'union__tweets__tfidfvect__smooth_idf': True, 'union__tweets__tfidfvect__ngram_range': (1, 2),
 'union__metadata__selectKbest__k': 0, 'union__desc__tfidfvect__analyzer': 'word',
 'union__metadata__scale__with_std': True, 'union__desc__tfidfvect__smooth_idf': False}


SVC
Best: 0.832572 using
{'union__tweets__tfidfvect__min_df': 1, 'model__kernel': 'linear',
 'union__tweets__tfidfvect__sublinear_tf': True,
 'union__tweets__tfidfvect__smooth_idf': True, 'model__tol': 1e-06, 'model__C': 1.0,
 'union__tweets__tfidfvect__use_idf': False, 'union__metadata__selectKbest__k': 0,
 'union__tweets__tfidfvect__max_df': 0.9}

SVC with user description
Best: 0.842834 using
{'union__tweets__tfidfvect__min_df': 5, 'union__transformer_weights': {'tweets': 1, 'desc': 0.4},
 'union__tweets__tfidfvect__analyzer': 'word', 'union__desc__tfidfvect__min_df': 1,
 'model__kernel': 'rbf', 'union__tweets__tfidfvect__use_idf': True,
 'union__tweets__tfidfvect__max_df': 0.9, 'union__desc__tfidfvect__sublinear_tf': True,
 'union__desc__tfidfvect__use_idf': False, 'union__desc__tfidfvect__analyzer': 'char',
 'union__desc__tfidfvect__max_df': 0.9, 'union__tweets__tfidfvect__sublinear_tf': True,
 'model__tol': 1e-10, 'union__tweets__tfidfvect__smooth_idf': True,
 'union__desc__tfidfvect__smooth_idf': False, 'model__C': 35.0}


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

GradientBoosting without metadata
Best: 0.843227 using
{'model__reg_alpha': 0.0, 'union__tweets__tfidfvect__max_df': 0.9,
 'union__tweets__tfidfvect__smooth_idf': False, 'model__learning_rate': 0.01,
 'union__tweets__tfidfvect__use_idf': True, 'union__tweets__tfidfvect__ngram_range': (1, 1),
 'union__tweets__tfidfvect__sublinear_tf': True, 'model__max_depth': 3,
 'union__tweets__tfidfvect__min_df': 1, 'model__gamma': 0,
 'union__tweets__tfidfvect__analyzer': 'word', 'model__booster': 'gblinear',
 'model__reg_lambda': 1.0, 'model__n_estimators': 100}

"""
