"""
Train Fast Text model

Author: Adrian Ahne (AA)

Date: 22-10-2018
"""

import argparse
import pandas as pd
from gensim.models import FastText
import multiprocessing
import os.path as op
import sys

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
from mongoDB_utils import connect_to_database

load_library(op.join(basename, 'preprocess'))
from preprocess import Preprocess





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train FastText model to create word vectors. \
                                                  Two options: 1) local mode: specify MongoDB host, \
                                                  MongoDB port, MongoDB database, MongoDB collection; \
                                                  2) cluster mode: specify path to data",
                                     epilog='Example usage in local mode : \
                                             python FastText_train.py -m "local" \
                                                                        -lh "localhost" \
                                                                        -lp "27017" \
                                                                        -ldb "tweets" \
                                                                        -lc "en_noRetweets" \
                                                                        --iter 50 \
                                                                        --alpha 0.05 \
                                            ')
    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-lh", "--localMongoHost", help="Host to connect to MongoDB (default=localhost)", default="localhost")
    parser.add_argument("-lp", "--localMongoPort", help="Port to connect to MongoDB (default=27017)", default="27017")
    parser.add_argument("-ldb", "--localMongoDatabase", help="MongoDB database to connect to")
    parser.add_argument("-lc", "--localMongoCollection", help="MongoDB collection (table) in which data is stored")
    parser.add_argument("-lpar", "--localParquetfile", help="Path to the parquet file")
    parser.add_argument("-cip", "--columnsInParquet", help="Names of column to extract", default=None)
    parser.add_argument("-lcsv", "--localCSV", help="Path to the csv file")
    parser.add_argument("-lcsvS", "--localCSVDelimiter", help="Delimiter used in csv file (default=',')", default=",")
    parser.add_argument("-cp", "--clusterPathData", help="Path to the data in cluster mode")
    parser.add_argument("-dcn", "--dataColumnName", help="If data stored in tabular form, gives the column of the desired text data (default='tweetText')", default="tweetText")
    parser.add_argument("--vecDim", help="Vector dimension of the word embedding (default=200)", default=200, type=int)
    parser.add_argument("--window", help="Maximum distance between the current and predicted word within a sentence (default=5)", default=5, type=int)
    parser.add_argument("--minCount", help="The model ignores all words with total frequency lower than this (default=1)", default=1, type=int)
    parser.add_argument("--localWorkers", help="Number of worker threads to train the model (default=all possible cores of machine)", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("--sg", help="Training algo: Skip-gram if sg=1, otherwise CBOW (default=1)", choices=[0,1], default=1)
    parser.add_argument("--hs", help="Hierarchical softmax used for training if hs=1, otherwise negative sampling (default=0)", choices=[0,1], default=0)
    parser.add_argument("--alpha", help="Initial learning rate (default=0.025)", default=0.025)
    parser.add_argument("--seed", help="Seed for random number generator (default=1)", default=1)
    parser.add_argument("--iter", help="Number of iterations (epochs) over the corpus (default=20)", default=20, type=int)
    parser.add_argument("--word_ngrams", help="Uses enriched word vectors with subword information if 1, otherwise this is like Word2Vec if 0 (default=1)", default=1, choices=[0,1])
    parser.add_argument("--min_n", help="Minimum length of char n-grams to be used for training (default=3)", default=3, type=int)
    parser.add_argument("--max_n", help="Maximum length of char n-grams to be used for training (default=6)", default=6, type=int)
    parser.add_argument("-s", "--savePath", help="Path where to save model to", required=True)

    args = parser.parse_args()


    # Preprocessing class
    prep = Preprocess(lang="english")


    # get tweets
    if args.mode == "local":

        # check from which source to read the data
        # check if parquet file
        if args.localParquetfile is not None:
            print("Local mode: Read parquet file..")
            if args.columnsInParquet is None:
                raw_tweets = pd.read_parquet(args.localParquetfile, engine="pyarrow")
            else:
                col = args.columnsInParquet.split(",")
                raw_tweets = pd.read_parquet(args.localParquetfile, engine="pyarrow", columns=col)

            print("Tokenize tweets..")
            tweets = []
            for tweet in raw_tweets[args.dataColumnName].values:
                tweets.append(prep.tokenize(tweet))

        # check if csv
        elif args.localCSV is not None:
            print("Local mode: Read csv file..")
            raw_tweets = pd.read_csv(args.localCSV, sep=args.localCSVDelimiter)

            print("Tokenize tweets..")
            tweets = []
            for tweet in raw_tweets[args.dataColumnName].values:
                tweets.append(prep.tokenize(tweet))

        # Check if necessary arguments are given
        elif args.localMongoDatabase is None and args.localMongoCollection is None:
            sys.stderr.write("ERROR: A MongoDB database and collection need to be provided to extract the data")
            sys.exit(1)

        else:
            print("Local mode: Connect to MongoDB collection..")
            client = connect_to_database()
            db = client[args.localMongoDatabase]
            collection = db[args.localMongoCollection]

            print("Tokenize tweets..")
            tweets = []
            for tweet in collection.find():
                tweets.append(prep.tokenize(tweet))

    elif args.mode == "cluster":

        # Check if necessary arguments are given
        if args.clusterPathData is None:
            sys.stderr.write("ERROR: A path to file containing the data needs to be provided")
            sys.exit(1)

        print("Cluster mode: Read parquet files..")
        # TODO: Load data from cluster and tokenize
        raw_tweets = pd.read_parquet(args.clusterPathData, engine="pyarrow")

        print("Tokenize tweets..")
        tweets = []
        for tweet in raw_tweets[args.dataColumnName].values:
            tweets.append(prep.tokenize(tweet))



    else:
        print("ERROR: Provided mode : {} is not supported. Possible options (local, cluster) ".format(args.mode))
        sys.exit()

    import ipdb; ipdb.set_trace()


    print("Train FastText...")
    model_ft = FastText(tweets, size=args.vecDim, window=args.window, min_count=args.minCount,
                        workers=args.localWorkers ,sg=args.sg, hs=args.hs, iter=args.iter,
                        word_ngrams=args.word_ngrams, min_n=args.min_n, max_n=args.max_n,
                        seed=args.seed, alpha=args.alpha)

    print("Save model to disk...")
    #file_name = "Trained_FastText_{}.model".format(datetime.datetime.now().strftime(DATE_FORMAT))
    model_ft.save(args.savePath)
