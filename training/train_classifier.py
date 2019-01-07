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
import argparse
#from pymongo import MongoClient
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
#from xgboost import XGBClassifier
import datetime
import gensim
from gensim.models import FastText
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing
import os.path as op
import json


# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
#from mongoDB_utils import connect_to_database
from tweet_utils import *
#from keras_utils import *


# CONSTANTS
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"




if __name__ == '__main__':



    parser = argparse.ArgumentParser(description="Train user classifier (personal or institutional) ",
                                     epilog='Example usage in local mode : \
                                             python train_classifier_user.py -m "local"  \
                                             -pwe "PAAAATH" -twe "ft" -ptd "PAAATH_TRAINING" \
                                             -mo "SVC", \
                                             --parameterGrid {"model__kernel":["linear", "poly", "rbf"], \
                                              "model__C":  [ 12.0, 10.0, 8.0, 6.0, 4.0, 1.0]} \
                                            ')
    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-pwe", "--pathWordEmbedding", help="Path to the word embeddings", required=True)
    parser.add_argument("-twe", "--typeWordEmbedding", help="FastText or Word2Vec embedding (default: ft)", choices=["ft", "w2v"], default="ft")
    parser.add_argument("-ptd", "--pathTrainingSet", help="Path to the training data csv", required=True)
    parser.add_argument("-ln", "--columnNameLabel", help="Column name of the label", default="personal (0=no, 1=yes)")
    parser.add_argument("-tn", "--columnNameTextData", help="Column name of the text data", default="tweet")
    parser.add_argument("-pg", "--parameterGrid", help="Parameter grid", type=json.loads, default={})
    parser.add_argument("-mo", "--modelAlgo", help="Trainings algorithm",
                        choices=["SVC", "logReg", "RandomForest", "XGBoost", "MultinomialNB", "MLP"], default="SVC")
    parser.add_argument("-sm", "--savePathTrainedModel", help="Path where to save trained model", default= "best_model_classif_{}.sav".format(datetime.datetime.now().strftime(DATE_FORMAT)))
    parser.add_argument("-t", "--type", help="Type of execution, mode: ['gridsearch', 'bestmodel'] ; \n gridsearch: executes a gridsearch with the provided parameterGrid; \n bestmodel: trains a model for the given parameter and saves it", choices=["gridsearch", "bestmodel"], default="gridsearch")


    args = parser.parse_args()

    # load csv in which we manually labelled the users
    #path_tweets = "D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\data\\manual_labeled_tweets\\manually_labeled_users_instVSpers_MoreInstTweets_30072018.csv"
    #tweets_csv = pd.read_csv(path_tweets, sep=";",
    #                         converters={"tweet_proc": lambda x: x.strip("[]").replace("'", "").split(", ")})

    print("Read training set ..")
    path_tweets = args.pathTrainingSet
    tweets_csv = pd.read_csv(path_tweets, sep=";")


    # get labels and tweet matrix
    labels = tweets_csv[args.columnNameLabel]
    tweets_raw = tweets_csv[args.columnNameTextData]

    print("Load word vectors..")
    if args.typeWordEmbedding == "ft":
        try:
            model_we = FastText.load(args.pathWordEmbedding)
        except:
            try:
                # downloaded pretrained word embeddings
                model_we = gensim.models.wrappers.FastText.load_fasttext_format(args.pathWordEmbedding)
            except NotImplementedError as error :
                print(error)
                print("The given type word embedding is not defined: {}".format(args.pathWordEmbedding))
    elif args.typeWordEmbedding == "w2v":
        try:
            model_we = Word2Vec.load(args.pathWordEmbedding)
        except:
            try:
                # downloaded pretrained word embeddings
                model_we = gensim.models.KeyedVectors.load_word2vec_format(args.pathWordEmbedding, binary=False)
            except NotImplementedError as error :
                print(error)
                print("The given type word embedding is not defined: {}".format(args.pathWordEmbedding))

    else:
        print("ERROR: Unknown type wordEmbedding: {}".format(args.typeWordEmbedding))



    
    # Adapt preprocessing function depending on which preprocessed word embeddings you use!
    V = np.array([tweet_vectorizer(preprocess_tweet(tweet, mode="no_preprocessing"), model_we) for tweet in tweets_raw])


    # remove the tweets that are empty because there is no word embedding
    VV = [(tweet, label) for tweet, label in zip(V, labels) if len(tweet) > 0]

    labels = pd.DataFrame([tup[1] for tup in VV])
    V = [tup[0] for tup in VV]


    # choose algo:
    #---------------------------------------------------------------------------
    if args.modelAlgo == "MultinomialNB":
        model = MultinomialNB(random_state=0)
    elif args.modelAlgo == "SVC":
        model = SVC(random_state=0)
    elif args.modelAlgo == "logReg":
        model = LogisticRegression(random_state=0)
    elif args.modelAlgo == "RandomForest" :
        model = RandomForestClassifier(random_state=0)
    elif args.modelAlgo == "XGBoost" :
        model = XGBClassifier(random_state=0)
    elif args.modelAlgo == "MLP" :
        model = MLPClassifier(early_stopping=True, batch_size=32, random_state=0)


    pipeline  = Pipeline([
                    ('model', model),
                ])


    # parameter grid for grid search by using fastText embeddings
    parameters = args.parameterGrid

    # parameters_ft = {
    #               # param for MultinomialNB
    #               #'model__alpha': (10, 5, 1, 0.5, 0.1),
    #
    #               # param for LogisticRegression
    #               #'model__C' : [5, 3, 1.0, 0.8, 0.5, 0.1],
    #               #'model__tol' : [1e-10, 1e-9],
    #
    #               # param for SVC
    #               'model__kernel' : ["linear", "poly", "rbf"],
    #               'model__C' : [ 12.0, 10.0, 8.0, 6.0, 4.0, 1.0],
    #               'model__tol' : [1e-1, 1e-2, 1e-3],
    #
    #               # param for RandomForestClassifier
    #               #'model__n_estimators' : [70, 80, 100, 120],
    #               #'model__criterion' : ['gini', 'entropy'],
    #               #'model__max_features' : ['auto', 'log2'],
    #               #'model__max_depth' : [ 8, 10, 20]
    #
    #               # param for XGBoost Best: 0.910828 using {'model__learning_rate': 0.05, 'model__reg_alpha': 0, 'model__max_depth': 3, 'model__reg_lambda': 1.5, 'model__n_estimators': 300}
    #               #'model__max_depth' : [3,4],
    #               #'model__learning_rate' : [0.03, 0.05, 0.07],
    #               #'model__booster' : ["gblinear"], #["gbtree", "gblinear", "dart"],
    #               #'model__gamma' : [0, 0.01],
    #               #'model__n_estimators' : [200, 300, 400, 500],
    #               #'model__reg_alpha' : [0, 0.1],
    #               #'model__reg_lambda' : [0.5, 1.0, 1.5]
    #
    #               # param for Multi layer perceptron
    #               #'model__hidden_layer_sizes' : [(64,64), (64,32)],#[(64), (64, 32), (32, 32)],
    #               #'model__activation' : ['relu', 'tanh', 'logistic'],
    #               #'model__solver' : ['adam', 'sgd'],
    #               #'model__learning_rate' : ['constant', 'invscaling'],
    #               #'model__tol' : [1e-2, 1e-3],#[1e-2, 1e-3, 1e-4],
    #
    #               #'model__hidden_layer_sizes' :  [(64), (32), (16,16)],#[(64), (16, 16), (32, 16)],
    #               #'model__activation' : ['relu'],# ['relu', 'tanh'],
    #               #'model__solver' : ['adam'],#['adam', 'sgd'],
    #               #'model__learning_rate' : ['constant', 'invscaling'],
    #               #'model__tol' : [1e-2, 1e-3, 1e-4],
    #               #'model__alpha' : [ 1e-4, 1e-5, 1e-6],
    #               #'model__max_iter' : [200, 300],
    #               #'model__beta_1' : [0.990, 0.999],
    #               #'model__beta_2' : [1e-7, 1e-8, 1e-9]
    #
    # }


    # Two options:
    #   1) Grid search to find best model
    #   2) Train best model and save to disk

    # Option 1) Grid search to find best model
    if args.type == "gridsearch":
        print("Start Grid search...")
        grid = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=2)

        grid = grid.fit(V, labels.values.ravel())

        #print(grid.cv_results_)
        print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))
    
    elif args.type == "bestmodel":

        # Option 2) Train best model and save to disk
        print("Train best model:")
        # train best model
#        best_model = SVC()  # training acc of 90.20%; no preprocessing
#        best_params = {
#                  'model__tol': 0.1,
#                  'model__C' : 10.0,
#                  'model__kernel':'rbf',
#                  'model__gamma': 0.1
#        }

        # create best pipeline
        best_pipeline = pipeline
        best_pipeline.set_params(**args.parameterGrid)

        print("Train model...")
        best_pipeline_trained = best_pipeline.fit(V, labels.values.ravel())

        # save best model
        print("Save model to {} ...".format(args.savePathTrainedModel))
    #    file_name = "best_model_user_classif_{}.sav".format(datetime.datetime.now().strftime(DATE_FORMAT))
        joblib.dump(best_pipeline_trained, args.savePathTrainedModel)

    else:
        print("ERROR: Provided type {} is not implemented!".format(args.type))
    


"""
RESULTS FROM AUGUST 2018!!

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
