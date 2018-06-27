"""

Author: Adrian Ahne
Date: 27-06-2018

Creates a collection consisting only of english and original tweets. All retweetes
or dubplicates are excluded

"""

from pymongo import MongoClient
from pprint import pprint
import numpy as np
import scipy
import pandas as pd
import sys
import pymongo
from bson.objectid import ObjectId




def add_noRetweets_to_collection(english_tweets, english_noRetweet_tweets):
    """
        Add all tweets that are no retweets

        Parameters:
          - english_tweets : collection of all tweets in english language
          - english_noRetweet_tweets : new collection in which only original
                                       english tweets are stored (no retweets!)

        REMARK: retweet that refers to oldest original tweet dates from 2008!)

    """
    for tweet in english_tweets.find( { "retweeted_status": { "$exists" : False }} ):

        try:
            english_noRetweet_tweets.insert_one(tweet)
        except Exception as e:
            print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))



def add_originalOfRetweet_to_collection(english_tweets, english_noRetweet_tweets):
    """
        Add the original tweet of a retweet to the english_noRetweet_tweets collection

        Parameters:
          - english_tweets : collection of all tweets in english language
          - english_noRetweet_tweets : collection to which we add the original
                                       tweet of a retweet
    """

    for tweet in english_tweets.find( { "retweeted_status": { "$exists" : True }} ):
        tweet_orig = tweet['retweeted_status']
        tweet_orig['number_of_weeks'] = tweet['number_of_weeks']

        try:
            english_noRetweet_tweets.insert_one(tweet_orig)
        except Exception as e:
            print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))



def get_duplicates(english_noRetweet_tweets):
    """
        Get all tweets that are duplicates (identic)
        Returns a dictionary with the lists of id's of all tweets that are
        copies of one same tweet
    """
    duplicates = english_noRetweet_tweets.aggregate( [
        { "$group": {
            # Group by fields to match on (a,b)
            "_id": { "id_str": "$id_str" },

            # Count number of matching docs for the group
            "count": { "$sum":  1 },

            # Save the _id for matching docs
            "docs": { "$push": "$_id" }
        }},

        # Limit results to duplicates (more than 1 match)
        { "$match": {
            "count": { "$gt" : 1 }
        }}
    ], allowDiskUse=True)

    return duplicates


def delete_duplicates(duplicates, english_noRetweet_tweets):
    """
        Deletes all duplicates / identic tweets

        Parameters:
          - duplicates : dict of lists of copies / identical tweets
          - english_noRetweet_tweets : collection from which duplicates are deleted
    """

    for duplicate in duplicates:
        n = len(duplicate['docs']) # number of copies of the current tweet

        # delete each duplicate
        i=1
        while i < n:
            result = client.tweets_database.filtered_tweets_noRetweets_english.delete_one({'_id': ObjectId(duplicate['docs'][i])})
            print(result.deleted_count)
            i += 1



if __name__ == '__main__':

    # connect to MongoDB database
    try:
        client = MongoClient('localhost', 27017) # host, port
    except ConnectionFailure as e:
        sys.stderr.write("Could not connect to MongoDB: %s" % e)
        sys.exit(1)

    # get database with all tweets
    db = client.tweets_database

    # new database with cleaned tweets (without Retweets, in english)
    english_tweets = db.english_tweets

    english_noRetweet_tweets = db.english_noRetweet_tweets

    #print("All collections in the database:")
    #print(db.collection_names())

    print("Number of english tweets:", english_tweets.count())

    # add english, original tweets to collection (no retweets in this collection )
    add_noRetweets_to_collection(english_tweets, english_noRetweet_tweets)
    print("Number of english, noRetweet tweets:", english_noRetweet_tweets.count())

    # add original tweet of retweeted tweet to collection
    add_originalOfRetweet_to_collection(english_tweets, english_noRetweet_tweets)

    # get all duplicate tweets (copies of tweets)
    duplicates = get_duplicates(english_noRetweet_tweets)

    # delete duplicates
    delete_duplicates(duplicates, english_noRetweet_tweets)
