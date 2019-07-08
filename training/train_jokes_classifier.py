import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier
import datetime
import gensim
from gensim.models import FastText
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import os.path as op
import json
from imblearn.pipeline import Pipeline


# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
from tweet_utils import *
from sklearn_utils import ItemSelect


def load_wordembeddings(path, type_embedding="ft"):
    if type_embedding == "ft":
        try:
            return FastText.load(path)
        except:
            try:
                # downloaded pretrained word embeddings
                return gensim.models.wrappers.FastText.load_fasttext_format(args.pathWordEmbedding)
            except NotImplementedError as error :
                print(error)
                print("The given type word embedding is not defined: {}".format(args.pathWordEmbedding))
    elif args.typeWordEmbedding == "w2v":
        try:
            return Word2Vec.load(args.pathWordEmbedding)
        except:
            try:
                # downloaded pretrained word embeddings
                return gensim.models.KeyedVectors.load_word2vec_format(args.pathWordEmbedding, binary=False)
            except NotImplementedError as error :
                print(error)
                print("The given type word embedding is not defined: {}".format(args.pathWordEmbedding))

    else:
        print("ERROR: Unknown type wordEmbedding: {}".format(args.typeWordEmbedding))
        sys.exit(1)


def preprocess_tweet(tweet):
    tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="replace", mode_Mentions="replace")
    tweet = prep.tokenize(tweet)
    return tweet


def get_model(modelAlgo):

    if modelAlgo == "MultinomialNB":
        return MultinomialNB(random_state=0)
    elif modelAlgo == "SVC":
        return SVC(random_state=0)
    elif modelAlgo == "logReg":
        return LogisticRegression(random_state=0)
    elif modelAlgo == "RandomForest" :
        return RandomForestClassifier(random_state=0)
    elif modelAlgo == "XGBoost" :
        return XGBClassifier(random_state=0)
    elif modelAlgo == "MLP" :
        return MLPClassifier(early_stopping=True, batch_size=32, random_state=0)
    else:
        print("ERROR: Modelname {} not provided!".format(modelname))
        sys.exit(1)


def get_pipeline():
    pipeline  = Pipeline([
                    ('union', FeatureUnion(
                                transformer_list = [
                                    ('tweet', Pipeline([
                                        ('tweetsSelector', ItemSelect(key='text')),
                                    ])),
                                    ('userDesc', Pipeline([
                                        ('userDescSelector', ItemSelect(key='user_description'))
                                    ])),
                                ],
                    )),
                    ('smote', SMOTE(random_state=12, ratio = 1.0, n_jobs=-1)),
                    ('model', model),
                ])

    return pipeline

if __name__ == '__main__':



    parser = argparse.ArgumentParser(description="Train jokes classifier (joke or noJoke) ",
                                     epilog='Example usage in local mode : \
                                             python train_jokes_classifier.py -m "local"  \
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
    parser.add_argument("-pg", "--parameterGrid", help="Parameter grid", type=json.loads, default={})
    parser.add_argument("-mo", "--modelAlgo", help="Trainings algorithm",
                        choices=["SVC", "logReg", "RandomForest", "XGBoost", "MultinomialNB", "MLP"], default="SVC")
    parser.add_argument("-sm", "--savePathTrainedModel", help="Path where to save trained model", default=None)
    parser.add_argument("-sc", "--scoring", help="Scoring functions for gridsearch", default=None )

    args = parser.parse_args()


    print("Read training set ..")
    path_tweets = args.pathTrainingSet
    tweets_csv = pd.read_csv(path_tweets, sep=";")

    print("Load word vectors..")
    model_we = load_wordembeddings(args.pathWordEmbedding, args.typeWordEmbedding)

    label = args.columnNameLabel
    data_pd = trainingData[["text", "user_description", label]]

    data_pd.text = data_pd.text.map(lambda tweet: tweet_vectorizer(preprocess_tweet(tweet), model_we))
    data_pd.user_description = data_pd.user_description.map(lambda userDesc: np.zeros((300, ))
                                                if isinstance(userDesc, float) or userDesc == " "
                                                else tweet_vectorizer(preprocess_tweet(userDesc), model_we))

    # remove the tweets that are empty because there is no word embedding
    data_pd = data_pd[data_pd["text"].apply(lambda x: len(x) > 0) ]
    print("Size training data:", data_pd.shape)

    model = get_model(args.modelAlgo)
    pipeline = get_pipeline()

    X = data_pd[["text", "user_description"]]
    y = data_pd[label]
    print("y.unique: ", y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_pd = pd.DataFrame(X_train, columns=["text", "user_description"])
    X_test_pd = pd.DataFrame(X_test, columns=["text", "user_description"])


    print("Start Grid search...")
    grid = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=1, scoring=args.scoring)
    grid.fit(X_train_pd, y_train)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))

    y_pred = grid.best_estimator_.predict(X_test_pd)
    print("F1-Score:", f1_score(y_test, y_pred))
    print("Precision: ",precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("roc auc: ", roc_auc_score(y_test, y_pred))
    print("Performance overall: ")
    print(classification_report(y_test, y_pred))

    if args.savePathTrainedModel != None:
        joblib.dump(grid.best_estimator_, args.savePathTrainedModel)
