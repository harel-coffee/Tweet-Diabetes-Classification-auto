"""
Author: Adrian Ahne
Date: 21-08-2018

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


def preprocess_tweet(tweet):
    tweet = prep.replace_contractions(tweet)
    tweet = prep.replace_special_words(tweet)
    tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="delete",
                                           mode_Hashtag="replace")
    tweet = prep.tokenize(tweet)
    tweet = prep.remove_punctuation(tweet)

    tweet = prep.preprocess_emojis(tweet)
    tweet = prep.preprocess_emoticons(tweet)
    tweet = prep.remove_non_ascii(tweet)
    tweet = prep.replace_numbers(tweet)
    tweet = prep.to_lowercase(tweet)

    tweet = prep.remove_stopwords(tweet, include_personal_words=False, include_negations=False)
    tweet = prep.lemmatize_verbs(tweet)
    tweet = prep.stem_words(tweet)
    #tweet = [ x for x in tweet if x not in[""] ]
    tweet = [ x for x in tweet if x not in["diabet", ""] ]

    return tweet



def concatenateTweetsOfMonthToDoc(users, collection):
    """
        Concatenate all tweets of one user in a month to a single document

        Parameters
        ----------------------------------------------------------------------
        users:      list of unique users of the collection
        collection: MongoDB collection with all the tweets

        Return
        ----------------------------------------------------------------------
        tweet_docs:         list with tweets concatenated in docs
        tweet_docs_prep:    list with preprocessed tweets concatenated in docs
    """
    count_single_tweets = 0
    for user in users:
        user_tweets = en_noRetweets_noInstitutions.find({'user.screen_name' : user})
        if user_tweets.count() == 1:
            count_single_tweets += 1

    print("Number of users that tweet only once:", count_single_tweets)


    tweet_docs = []
    tweet_docs_prep = []
    for j, user in enumerate(users):
        user_tweets = en_noRetweets_noInstitutions.find({'user.screen_name' : user})
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


            tweet_docs.append(tweet_doc)
            tweet_docs_prep.append(preprocess_tweet(tweet_doc))

        if j %20000 == 0:
            print(j)

    return tweet_docs, tweet_docs_prep


def to_date(twitter_date):
    return datetime.datetime.strptime(twitter_date, '%a %b %d %H:%M:%S +0000 %Y')


def gridsearch_graph(dictionary, corpus, texts, list_num_topics):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : preprocessed tweets
    list_num_topics: list with number of topics to calculate the LDA on

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    logPerplex_list = []
    for num_topics in list_num_topics:
        print("number of topics:", num_topics)
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=0,
                      chunksize=5000, passes=50, eval_every=None, alpha='auto', eta='auto', iterations=50)


        lm_list.append(lm)
        logPerplex_list.append(lm.log_perplexity(corpus))
        cm = CoherenceModel(model=lm, corpus=corpus, dictionary=dictionary, texts=texts, coherence='c_v', processes=-1)
        c_v.append(cm.get_coherence())

    # Show graph
    #x = list_num_topics #range(1, limit)
    #plt.plot(x, c_v)
    #plt.xlabel("num_topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("c_v"), loc='best')
    #plt.show()

    return lm_list, c_v, logPerplex_list



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

    # add path to utils directory to system path
    path = 'D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\\utils'
    if path not in sys.path:
        sys.path.insert(0, path)

    from sys_utils import *
    from mongoDB_utils import *

    # load preprocess function
    load_library('D:\A_AHNE1\Tweet-Classification-Diabetes-Distress\preprocess')
    from preprocess import Preprocess
    prep = Preprocess()

    print("Connect to database...")
    client = connect_to_database(host='localhost', port=27017)
    # get database with all tweets
    db = client.tweets_database
    en_noRetweets_noInstitutions = client.tweets_database.en_noRetweets_noInstitutions


    print("Get distinct users...")
    en_noRetweets_noInstitutions.create_index("user.screen_name")
    users = en_noRetweets_noInstitutions.find().distinct("user.screen_name")

    print("Number of distinct users:", len(users))
    print("Number of tweets:", en_noRetweets_noInstitutions.count())

    print("Concatenate all tweets of one person in one month to one single document...")
    tweet_docs, tweet_docs_prep = concatenateTweetsOfMonthToDoc(users, en_noRetweets_noInstitutions)

    print("Number of documents (concatenated tweets): ", len(tweet_docs))

    print("Create dictionary...")
    dictionary = Dictionary(tweet_docs_prep)

    print("Create bag of words...")
    corpus = [dictionary.doc2bow(text) for text in tweet_docs_prep]

    list_num_topics = [6, 8, 10, 12, 14]
    print("Calculate LDA...")
    lmlist, c_v, logPerplex = calculateLDA(dictionary=dictionary, corpus=corpus,
                                texts=tweet_docs_prep, list_num_topics=list_num_topics,
                                saveModelPath="D:\\A_AHNE1\\Tweet-Classification-Diabetes-Distress\\topicModel\\trained_LDA__20-08-2018_excludedWords\\")
#    lmlist, c_v, logPerplex_list = gridsearch_graph(dictionary=dictionary, corpus=corpus,
#                                texts=tweet_docs, list_num_topics=list_num_topics)


    import ipdb; ipdb.set_trace()


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

    plt.show()

    import ipdb; ipdb.set_trace()
