from numpy.linalg import norm
from gensim.models import FastText
import sys
import os.path as op
import random
import pandas as pd
import numpy as np
import argparse



basename = "/home/adrian/PhD/Tweet-Classification-Diabetes-Distress/"
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
from tweet_utils import tweet_vectorizer

from preprocess import Preprocess
prep = Preprocess()



def kmeans(data, k, maxIterations, distance="cosinus", vectorColumn="text_vec"):

    # Initialize centroids randomly
    centroids = initialise_centroids(np.asarray(data[vectorColumn].values.tolist()), k)   # [ arr, arr, .., arr ]

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = [np.zeros((300,)) for i in range(k)]

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIterations):
        print("Iterations:", iterations)
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        data["label"] = data[vectorColumn].map(lambda d: getLabelForEachDataPoint(d, centroids, distance=distance)).astype('category')
        #labels = getLabelForEachDataPoint(data, centroids)

        # Assign centroids based on datapoint labels
        centroids = data.groupby(by="label").apply(lambda value: getMean(value, vectorColumn))#.mean()

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return data


def getMean(data, vectorColumn):
    matrix_form = np.asarray(data[vectorColumn].values.tolist())
    return np.sum(matrix_form,axis=0) / matrix_form.shape[0]


def updateCentroids(data, labels, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.

    # recalculate centroids
    index = 0
    for cluster in labels:
        old_centroids[index] = data[index]
        data[index] = np.mean(cluster, axis=0).tolist()
        index += 1

def initialise_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    return [data[i] for i in random_indices]


def shouldStop(oldCentroids, centroids, iterations, maxIterations):
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
    if iterations > maxIterations: return True
#    return all([(centroids[i] == oldCentroids[i]).all() for i in range(len(centroids))])#(oldCentroids == centroids).all()
    return all([np.array_equal(centroids[i], oldCentroids[i]) for i in range(len(centroids))])#(oldCentroids == centroids).all()

def cosinus_similarity(a, b):
    return np.inner(a,b)/(norm(a)*norm(b))


# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.
def getLabelForEachDataPoint(tweet_vec, centroids, distance="cosinus"):

    if distance == "cosinus": return max([(i, cosinus_similarity(tweet_vec, center)) for i, center in enumerate(centroids)], key=lambda t: t[1])[0]
    elif distance == "euclidean" : return min([(i, np.linalg.norm(tweet_vec-center))for i, center in enumerate(centroids)], key=lambda t: t[1])[0]
    else: print("ERROR: Wrong distance meausure!")


def preprocess_tweet(tweet):
    tweet = prep.replace_contractions(tweet)
    tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="delete")
    tweet = prep.remove_repeating_characters(tweet)
    tweet = prep.remove_repeating_words(tweet)
    tweet = prep.tokenize(tweet)
    tweet = prep.remove_punctuation(tweet)
    tweet = prep.to_lowercase(tweet)
    tweet = prep.remove_non_ascii(tweet)
    tweet = prep.replace_numbers(tweet)
    tweet = prep.remove_stopwords(tweet, include_personal_words=False, include_negations=False)
    tweet = [word for word in tweet if word not in ["diabetes", "diabetic"]]
    return tweet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run kmeans clustering \
                                     epilog='Example usage in local mode : \
                                             python kmeans.py -fn "..." ... \
    parser.add_argument("-fn", "--filename", help="Path to the data file")
    parser.add_argument("-we", "--wordEmbedding", help="Path to word embeddings")
    parser.add_argument("-lfd", "--filenameDelimiter", help="Delimiter used in file (default=',')", default=",")
    parser.add_argument("-lfc", "--filenameColumns", help="String with column names")
    parser.add_argument("-dcn", "--dataColumnName", help="If data stored in tabular form, gives the column of the desired text data (default='tweetText')", default="text")
    parser.add_argument("-maxIt", "--maxIterations", help="Maximum number of iterations of kmeans", default=150)
    parser.add_argument("-N", "--Ncluster", help="Number of clusters", required=True)
    parser.add_argument("-dist", "--distance", help="distance measure", default="cosinus")
    parser.add_argument("-s", "--savePath", help="Path where to save model to", required=True)

    args = parser.parse_args()

    print("Load data..")
    data = readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter)

    print("Load word embeddings..")
    model_ft = FastText.load(args.wordEmbedding)

    print("Preprocess data..")
    data["text_vec"] = data[args.dataColumnName].map(lambda tweet: tweet_vectorizer(preprocess(tweet), model))
    data["prep"] = data[args.dataColumnName].map(lambda tweet: preprocess(tweet))


    Nclusters = [20, 30]

    for N in Nclusters:
        res=kmeans(data, Ncluster, args.maxIterations, distance=args.distance, vectorColumn="text_vec")
