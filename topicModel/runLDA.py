"""
Author: Adrian Ahne
Date: 12-03-2019

Topic model with Latent Dirichlet Allocation (LDA)
For a given number of clusters, calculate the topics
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import gensim
import re
import sys
import argparse
import os
import os.path as op
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from dateutil.relativedelta import relativedelta
import datetime
import matplotlib.pyplot as plt

basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library, strInput2bool
#from mongoDB_utils import connect_to_database

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
load_library(op.join(basename, 'preprocess'))
from defines import ColumnNames as cn
from defines import Patterns

load_library(op.join(basename, 'readWrite'))
from readWrite import readFile


def to_date(twitter_date):
    return datetime.datetime.strptime(twitter_date, '%a %b %d %H:%M:%S +0000 %Y')

def current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

def preprocess_tweet(tweet):
    tweet = prep.replace_contractions(tweet)
    tweet = prep.replace_special_words(tweet) # normalise "#type1", "Type 1", "t1d" to "type1"
    tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="delete",
                                           mode_Hashtag="replace")
    tweet = prep.remove_repeating_characters(tweet)
    tweet = prep.remove_repeating_words(tweet)
    tweet = prep.tokenize(tweet)
    tweet = prep.remove_punctuation(tweet)

    tweet = prep.preprocess_emojis(tweet, limit_nEmojis=3)
    tweet = prep.preprocess_emoticons(tweet)
    tweet = prep.remove_non_ascii(tweet)
    tweet = prep.replace_numbers(tweet)
    tweet = prep.to_lowercase(tweet)

    tweet = prep.remove_stopwords(tweet, include_personal_words=False, include_negations=False)
    tweet = prep.lemmatize_verbs(tweet)
    tweet = prep.stem_words(tweet)
    tweet = [ x for x in tweet if x not in["diabet", ""] ]

    return tweet



def concatenateTweetsOfMonthToDoc(users, data, format, user_screen_name):
    """
        Concatenate all tweets of one user in a month to a single document

        Parameters
        ----------------------------------------------------------------------
        users:      list of unique users of the collection
        data:       Tweets data
        format:     ["MongoDB", "Pandas] ; give type of data

        Return
        ----------------------------------------------------------------------
        tweet_docs:         list with tweets concatenated in docs
        tweet_docs_prep:    list with preprocessed tweets concatenated in docs
    """

    if format == "MongoDB":
        count_single_tweets = 0
        for user in users:
            user_tweets = data.find({'user.screen_name' : user})
            if user_tweets.count() == 1:
                count_single_tweets += 1

        print("Number of users that tweet only once:", count_single_tweets)


        tweet_docs = []
        tweet_docs_prep = []
        for j, user in enumerate(users):
            user_tweets = data.find({'user.screen_name' : user})
            nTweets = user_tweets.count()

            # if a user tweeted a lot, show how many
            if nTweets > 500:
                print("\tnTweets:", nTweets)

            if nTweets == 1:
                tweet = user_tweets.next()
                tweet_docs.append(tweet["text"])
                tweet_docs_prep.append(preprocess_tweet(tweet["text"]))

            # if a user tweeted several times, sort tweets
            else:
                user_tweets = user_tweets.sort('date', pymongo.ASCENDING)
                start_date = user_tweets[0]['created_at']
                end_date = to_date(start_date) + relativedelta(months=1)


                tweet_doc = user_tweets[0]["text"] # concatenation of tweets of a user of the same months
                nt = 1

                while nt < nTweets:

                    current_tweet = user_tweets[nt]

                    if to_date(current_tweet["created_at"]) < end_date:
                        tweet_doc += " "+current_tweet["text"]
                    else:
                        tweet_docs.append(tweet_doc)
                        tweet_docs_prep.append(preprocess_tweet(tweet_doc))

                        tweet_doc = current_tweet["text"]
                        end_date = to_date(current_tweet["created_at"]) + relativedelta(months=1)

                    nt += 1

                # don't add if preprocessed tweet is empty
                tweet_prep = preprocess_tweet(tweet_doc)
                if len(tweet_prep) >= 1 :
                    tweet_docs.append(tweet_doc)
                    tweet_docs_prep.append(tweet_prep)

            if j %20000 == 0:
                print(j)

        return tweet_docs, tweet_docs_prep

    elif format == "Pandas":
        count_single_tweets = 0
        for user in users:
            user_tweets = data.find({'user.screen_name' : user})
            if user_tweets.count() == 1:
                count_single_tweets += 1

        count_users_with_only_one_tweet = pd.Series(users).map(lambda user: data[data["user_screen_name"] == user].shape[0]).values.tolist().count(1)
        print("Number of users that tweet only once:", count_users_with_only_one_tweet)

        tweet_docs = []
        tweet_docs_prep = []
        for j, user in enumerate(users):
            user_tweets =  data[data["user_screen_name"] == user]
            nTweets = user_tweets.shape[0]

            # if a user tweeted a lot, show how many
            if nTweets > 500:
                print(user, "\tnTweets:", nTweets)

            if nTweets == 1:
                #tweet = user_tweets.iloc[0]["text"]
                tweet_docs.append(user_tweets.iloc[0]["text"])
                tweet_docs_prep.append(preprocess_tweet(user_tweets.iloc[0]["text"]))

            # if a user tweeted several times, sort tweets
            else:
                user_tweets["created_at"] = pd.to_datetime(user_tweets.created_at)
                user_tweets.sort_values(by='created_at', inplace=True, ascending=True)
                start_date = user_tweets.iloc[0]["created_at"]
                end_date = start_date + relativedelta(months=1)
                #user_tweets = user_tweets.sort('date', pymongo.ASCENDING)
                #start_date = user_tweets[0]['created_at']
                #end_date = to_date(start_date) + relativedelta(months=1)


                tweet_doc = user_tweets.iloc[0]["text"] # concatenation of tweets of a user of the same months
                nt = 1

                while nt < nTweets:

                    current_tweet = user_tweets.iloc[nt]

                    if to_date(current_tweet["created_at"]) < end_date:
                        tweet_doc += " "+current_tweet["text"]
                    else:
                        tweet_docs.append(tweet_doc)
                        tweet_docs_prep.append(preprocess_tweet(tweet_doc))

                        tweet_doc = current_tweet["text"]
                        end_date = to_date(current_tweet["created_at"]) + relativedelta(months=1)

                    nt += 1

                # don't add if preprocessed tweet is empty
                tweet_prep = preprocess_tweet(tweet_doc)
                if len(tweet_prep) >= 1 :
                    tweet_docs.append(tweet_doc)
                    tweet_docs_prep.append(tweet_prep)

            if j %20000 == 0:
                print(j)

        return tweet_docs, tweet_docs_prep
    else:
        sys.stderr.write("ERROR: Data for concatenation of all users of the same month should be either in format MongoDB or Pandas!")
        sys.exit(1)





def calculateLDA(dictionary, corpus, texts, list_num_topics, saveModelPath=[]):
    """
    Computes LDA models for given list with number of topics and save them to disk
    And calculates coherence values for each model

    Parameters:
    ----------
    dictionary:         Gensim dictionary
    corpus :            Gensim corpus
    texts :             Preprocessed  texts
    list_num_topics:    list with number of topics to find for LDA
    saveModelPath:      if empty, do nothing
                        otherwise save model to disk

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    logPerplex_list = []
    for num_topics in list_num_topics:
        print("\tNumber of topics:", num_topics)
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=0,
                      chunksize=5000, passes=50, eval_every=None, alpha='auto', eta='auto', iterations=50)


        lm_list.append(lm)
        cm = CoherenceModel(model=lm, corpus=corpus, dictionary=dictionary, texts=texts, coherence='c_v', processes=-1)
        logPerplex_list.append(lm.log_perplexity(corpus))
        c_v.append(cm.get_coherence())

        if saveModelPath != []:
            lm.save(saveModelPath+"K_{}.model".format(num_topics))

    print("Number topics:", list_num_topics)
    print("Coherence scores:", c_v)
    print("LogPerplexity: ", logPerplex_list)
    return lm_list, c_v, logPerplex_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run LDA",
                                     epilog='Example usage in local mode : \
                                             python filter.py -m "local" -lf "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/matching-tweets.parquet/Project=Diabetes" \
                                             -lfc "id, lang, text, user_screen_name, user_followers_count, \
                                              user_friends_count, user_location, user_description, user_tweets_count, \
                                              place_country, place_full_name,tweet_longitude, tweet_latitude, user_id, \
                                              retweeted_user_id, retweeted_user_screen_name, retweeted_user_followers_count, \
                                              retweeted_user_friends_count, retweeted_user_tweet_count, retweeted_user_location,\
                                              retweeted_user_description, retweeted_place_country, retweeted_place_full_name, \
                                              retweeted_tweet_longitude, retweeted_tweet_latitude, retweeted_text, is_retweet, posted_month" \
                                              -s "/space/Work/spark/matching-tweets_diabetes_noRetweetsDuplicates.parquet" \
                                              -wr "False" -wo "True" -cD {"retweeted_text":"retweeted_text"} \
                                              ')



    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-lh", "--localMongoHost", help="Host to connect to MongoDB (default=localhost)", default="localhost")
    parser.add_argument("-lp", "--localMongoPort", help="Port to connect to MongoDB (default=27017)", default="27017")
    parser.add_argument("-ldb", "--localMongoDatabase", help="MongoDB database to connect to")
    parser.add_argument("-lc", "--localMongoCollection", help="MongoDB collection (table) in which data is stored")
    parser.add_argument("-fn", "--filename", help="Path to the data file")
    parser.add_argument("-fnD", "--filenameDelimiter", help="Delimiter used in file (default=',')", default=",")
    parser.add_argument("-fnC", "--filenameColumns", help="String with column names")
    parser.add_argument("-s", "--saveResultPath", help="Path name where result should be stored")
    parser.add_argument("-K", "--numberTopics", help="Give the number of topics for the LDA as type string separated by a comma (ex: -k '10, 20, 50')", required=True)

    args = parser.parse_args()



#    path_save_LDA = "D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\topicModel\\emotion_LDA_11-09-2018_excludedWords\\"
#    path_save_dict = "D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\topicModel\\emotion_LDA_11-09-2018_excludedWords\\dictionary.dict"
    path_save_LDA = op.join(saveResultPath, "LDA_"+current_date()+"\\")#"D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\topicModel\\emotion_LDA_11-09-2018_excludedWords\\"
    path_save_dict = path_save_LDA+"dictionary.dict"

    print("Path to save LDA model to: ", path_save_LDA)
    print("Path to save LDA model to: ", path_save_dict)


    from preprocess import Preprocess
    prep = Preprocess()


    if args.mode == "local":

        # check from which source to read the data

        if args.filename is not None:
            print("Local mode: Read file..")

            df = readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter)

            print("Get distinct users...")
            unique_users = df.[user_screen_name].unique().tolist()
            print("Number of distinct users:", len(unique_users))

            print("Concatenate all tweets of one person in one month to one single document...")
            tweet_docs, tweet_docs_prep = concatenateTweetsOfMonthToDoc(unique_users, df, format="Pandas")


        # Check if necessary arguments are given
        elif args.localMongoDatabase is None and args.localMongoCollection is None:
            sys.stderr.write("ERROR: A MongoDB database and collection need to be provided to extract the data")
            sys.exit(1)

        # use MongoDB storage
        else:
            print("Local mode: Connect to MongoDB collection..")

            from mongoDB_utils import connect_to_database
            import pymongo
            from pymongo import MongoClient

            client = connect_to_database(host=args.localMongoHost, port=args.localMongoPort)
            # get database with all tweets
            db = client.[args.localMongoDatabase]

            db_collection = client.[args.localMongoCollection]

            print("Get distinct users...")
            db_collection.create_index("user.screen_name")
            users = db_collection.find().distinct("user.screen_name")

            print("Number of distinct users:", len(users))
            print("Number of tweets total:", db_collection.count())

            print("Concatenate all tweets of one person in one month to one single document...")
            tweet_docs, tweet_docs_prep = concatenateTweetsOfMonthToDoc(users, db_collection, format="mongoDB")
            print("Number of documents (concatenated tweets): ", len(tweet_docs))



        print("Create dictionary...")
        dictionary = Dictionary(tweet_docs_prep)
        print("Save dictionary (nTokens={}) to file {}...".format(len(dictionary.values()), path_save_dict))
        dictionary.save(path_save_dict)

        print("Create bag of words...")
        corpus = [dictionary.doc2bow(text) for text in tweet_docs_prep]

        list_num_topics = args.numberTopics.replace(" ", "").split(",")#[16, 20, 22, 24, 26, 28, 30]
        print("Calculate LDA...")
        lmlist, c_v, logPerplex = calculateLDA(dictionary=dictionary, corpus=corpus,
                            texts=tweet_docs_prep, list_num_topics=list_num_topics,
                            saveModelPath=path_save_LDA)


        print("Plot...")
        plt.figure()

        plt.subplot(121)
        plt.plot(list_num_topics, c_v)
        plt.xlabel("num_topics")
        plt.ylabel("Coherence score")
        plt.legend(("c_v"), loc='best')

        plt.subplot(122)
        plt.plot(list_num_topics, logPerplex)
        plt.xlabel("num_topics")
        plt.ylabel("log Perplexity")
        plt.legend(("logPerpl"), loc='best')

        plt.savefig(op.join(path_save_LDA, "plot_coherencescore_logPerplex.png"))

    elif mode == "cluster":
        sys.stderr.write("ERROR: Not implemented yet for cluster mode!")
        sys.exit(1)
