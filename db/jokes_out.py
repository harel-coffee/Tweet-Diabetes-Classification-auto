"""
AA
05-04-2019

Taking the personal tweets and filter out the jokes by using a classifier we trained before
"""
import argparse
from pprint import pprint
import numpy as np
import scipy
import pandas as pd
import sys
import os
import os.path as op
from sklearn.externals import joblib
from gensim.models import FastText

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
from tweet_utils import tweet_vectorizer
from sklearn_utils import ItemSelect

load_library(op.join(basename, 'readWrite'))
load_library(op.join(basename, 'preprocess'))
from preprocess import Preprocess
from readWrite import savePandasDFtoFile, readFile

prep = Preprocess()



def preprocess_tweet(tweet):
    try:
        tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="replace", mode_Mentions="replace")
        tweet = prep.tokenize(tweet)
        return tweet
    except:
        print("ERROR: can not preprocess tweet:", tweet)




def text_to_embedding(tweets, wordEmbedding, textColumn, userDescriptionColumn):
    temp = pd.DataFrame()
    temp[textColumn] = tweets[textColumn].map(lambda tweet: tweet_vectorizer(preprocess_tweet(tweet), wordEmbedding))
    temp[userDescriptionColumn] = tweets[userDescriptionColumn].map(lambda userDesc: np.zeros((200, ))
                                                if isinstance(userDesc, float) or userDesc == " " or userDesc is None
                                                else tweet_vectorizer(preprocess_tweet(userDesc), wordEmbedding))
    return temp[[textColumn, userDescriptionColumn]]

def filter_out_jokes(tweets, model_joke_classif, wordEmbedding, textColumn, userDescriptionColumn):
    """
        Filter out jokes based on the provided classifier
    """
    #return tweets[tweets.apply(lambda tweet:  classify_tweet(tweet[[textColumn,userDescriptionColumn]], wordEmbedding, textColumn, userDescriptionColumn, model_joke_classif) == 0, axis=1)]
    tweets_wordEmbedding = text_to_embedding(tweets, wordEmbedding, textColumn, userDescriptionColumn)
    return tweets[model_joke_classif.predict(tweets_wordEmbedding) == 0]





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Filter out jokes of provided database based on trained joke classifier ",
                                     epilog='Example usage in local mode : \
                                             python jokes_out.py -m "local"  \
                                             -puc "P_A_T_H" -ptc "P_A_T_H" -pwe "P_A_T_H" \
                                             -pdata "P_A_T_H" -sm 0.25 -s "P_A_T_H" \
                                      ')
    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-pc", "--pathJokeClassifier", help="Path to the joke classifier (joke vs non-joke tweet)", required=True)
    parser.add_argument("-pwe", "--pathWordEmbedding", help="Path to the word embeddings", required=True)
    parser.add_argument("-pdata", "--pathData", help="Path to the data to classify", required=True)
    parser.add_argument("-s", "--pathSave", help="Path to save cleaned database to (.parquet or .csv)")
    parser.add_argument("-ct", "--columnNameText", help="Column name of the text data", default="text")
    parser.add_argument("-cud", "--columnNameUserDescription", help="Column name of the user description data", default="user_description")

    args = parser.parse_args()

    print("Load joke classifier..")
    #loaded_model = joblib.load(filename)
    model_joke_classif = joblib.load(args.pathJokeClassifier)

    print("Load word embedding..")
    wordEmbedding = FastText.load(args.pathWordEmbedding)

    print("Load data..")
    tweets = readFile(args.pathData)
    print("Number tweets:", len(tweets))
    print(tweets.head())
    print("Apply joke classifier and filter out jokes..")
    tweets_noJokes = filter_out_jokes(tweets, model_joke_classif, wordEmbedding, args.columnNameText, args.columnNameUserDescription)
    print("Number no jokes tweets :", len(tweets_noJokes))
    print()
    print(tweets_noJokes.head(5))

    if args.pathSave != []:
        print("Save personal tweets to file {}  ...".format(args.pathSave))
        savePandasDFtoFile(tweets_noJokes, args.pathSave)
