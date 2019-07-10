import argparse
import pandas as pd
import numpy as np
import os.path as op
import unicodedata
import sys
from gensim.models import FastText
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, make_scorer

basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
from tweet_utils import *
from sklearn_utils import ItemSelect

from preprocess import Preprocess
prep = Preprocess()

def preprocess_tweet(tweet):
    tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="replace", mode_Mentions="replace")
    tweet = prep.tokenize(tweet)
    return tweet


def create_history_typeDiabetes_column(row):
    #print(row)
    if row["History_TypeDiab"] == 0: return 0
    elif row["History_TypeDiab"] == 1: return 1
    elif row["History_TypeDiab"] == 2: return 2
    elif pd.isnull(row["History_TypeDiab"]): return row["Type_Diabetes"]
    else: print("ERROR: Should not occur:  ", row["Type_Diabetes"], ";;;", row["Type_Diabetes"])


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
    from imblearn.pipeline import Pipeline
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
                    ('smote', SMOTE(random_state=12, ratio = "auto", n_jobs=-1)),
                    ('model', model),
                ])

    return pipeline

if __name__ == '__main__':



    parser = argparse.ArgumentParser(description="Train type of diabetes classifier (no diabetes = 0, type1 = 1, type2 = 2) ",
                                     epilog='Example usage in local mode : \
                                             python train_typeDiabetes_classifier.py  \
                                             -pwe "PAAAATH" -twe "ft" -ptd "PAAATH_TRAINING" \
                                             -mo "SVC", \
                                             --parameterGrid {"model__kernel":["linear", "poly", "rbf"], \
                                              "model__C":  [ 12.0, 10.0, 8.0, 6.0, 4.0, 1.0]} \
                                            ')
    parser.add_argument("-pwe", "--pathWordEmbedding", help="Path to the word embeddings", required=True)
    parser.add_argument("-twe", "--typeWordEmbedding", help="FastText or Word2Vec embedding (default: ft)", choices=["ft", "w2v"], default="ft")
    parser.add_argument("-ptd", "--pathTrainingSet", help="Path to the training data csv", required=True)
    parser.add_argument("-pg", "--parameterGrid", help="Parameter grid", type=json.loads, default={})
    parser.add_argument("-mo", "--modelAlgo", help="Trainings algorithm",
                        choices=["SVC", "logReg", "RandomForest", "XGBoost", "MultinomialNB", "MLP"], default="SVC")
    parser.add_argument("-sm", "--savePathTrainedModel", help="Path where to save trained model", default=None)
    parser.add_argument("-sc", "--scoring", help="Scoring functions for gridsearch", default=None )

    args = parser.parse_args()


    print("Read training set ..")
    path_tweets = args.pathTrainingSet
    trainingData = pd.read_csv(path_tweets)

    print("Load word vectors..")
    model_we = load_wordembeddings(args.pathWordEmbedding, args.typeWordEmbedding)

    # merge history Typediabetes column with normal type diabetes column
    trainingData['history_typeDiab_total'] = trainingData.apply (lambda row: create_history_typeDiabetes_column(row), axis=1)

    label = "history_typeDiab_total"
    data_pd = trainingData[["text", "user_description", "user_name", label]]

    print("Get word embeddings for tweet text, user description..")
    data_pd.text = data_pd.text.map(lambda tweet: tweet_vectorizer(preprocess_tweet(tweet), model_we))
    data_pd.user_description = data_pd.user_description.map(lambda userDesc: np.zeros((model_we.vector_size, ))
                                                if isinstance(userDesc, float) or userDesc == " "
                                                else tweet_vectorizer(preprocess_tweet(userDesc), model_we))

    # remove the tweets that are empty because there is no word embedding
    data_pd = data_pd[data_pd["text"].apply(lambda x: len(x)>0) ]
    print("data before filter out gestational diabetes:", data_pd.shape, type(data_pd))
    data_pd = data_pd.loc[data_pd[label] != 3]
    print("Size training data:", data_pd.shape)

    model = get_model(args.modelAlgo)
    pipeline = get_pipeline()

    X = data_pd[["text", "user_description"]]
    y = data_pd[label]
    print("y.unique: ", y.unique())
    print("Label counts:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_pd = pd.DataFrame(X_train, columns=["text", "user_description"])
    X_test_pd = pd.DataFrame(X_test, columns=["text", "user_description"])

    prec_scorer = make_scorer(precision_score, average="micro")

    print("Start Grid search...")
    grid = GridSearchCV(pipeline, args.parameterGrid, cv=10, n_jobs=-1, verbose=1, scoring=prec_scorer)
    grid.fit(X_train_pd, y_train)
    print("\nBest: %f using %s" % (grid.best_score_, grid.best_params_))

    y_pred = grid.best_estimator_.predict(X_test_pd)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Performance overall: ")
    print(classification_report(y_test, y_pred))

    if args.savePathTrainedModel != None:
        joblib.dump(grid.best_estimator_, args.savePathTrainedModel)
