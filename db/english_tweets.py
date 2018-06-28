"""

Author: Adrian Ahne
Date: 27-06-2018


Creates a collection consisting only of english tweets
Furthermore two fields are added to each tweet-document:
  - 'created_at_orig' : if tweet-document is no retweet -> insert date of the field 'created_at'
                        if tweet-document is retweet -> insert date of original tweet
                                                        of the field 'retweeted_status.created_at'

  - 'number_of_weeks' : Insert the number of week (int) the tweet is posted
                        based on 'created_at_orig'
                        Start date is 01-05-2017 00:00:00

The field 'number_of_weeks' will allow more precise analysis in a later step

"""


import pymongo
from pymongo import MongoClient
from pprint import pprint
import datetime



def create_temp_collection(raw_tweets, temp_tweets):
    """
        Add new field 'created_at_orig' which saves in datetime format the date of the tweet
        For retweets the date of the original post is saved

        Parameters:
          - raw_tweets: MongoDB collection of raw tweets
          - temp_tweets: MongoDB collection in which to store the tweets temporary
    """

    for tweet in raw_tweets.find({'lang': 'en'}):
        # if no retweet -> save date of tweet
        if "retweeted_status" not in tweet.keys():
            tweet['created_at_orig'] = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        # if retweet -> save date of original post
        else:
            tweet['created_at_orig'] = datetime.datetime.strptime(tweet['retweeted_status']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

        try:
            temp_tweets.insert_one(tweet)
        except Exception as e:
            print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))


def create_english_tweets_collection(temp_tweets, english_tweets, start_date):
    """
    Create database consisting only of english tweets and add additional field
    'number_of_weeks' to be able to do more precise analyses in a later step

    INFO:
    sort tweets by date (s.t. we can label them easily with the number of weeks and so to which bin they belong)

    """

    end_of_week = start_date + datetime.timedelta(days=7)
    number_of_week = 1

    # sort tweets by date and only greater than start_date
    for tweet in temp_tweets.find({'created_at_orig': {'$gt': start_date}}).sort('created_at_orig', pymongo.ASCENDING):

        if tweet['created_at_orig'] < end_of_week:
            tweet['number_of_weeks'] = number_of_week

            try:
                english_tweets.insert_one(tweet)
            except Exception as e:
                print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))

        else:
            # go to next week if current tweet was not in the current week bin
            number_of_week = number_of_week + 1
            end_of_week = end_of_week + datetime.timedelta(days=7)
            print("number of weeks:", number_of_week, " end of week: ", end_of_week)
            tweet['number_of_weeks'] = number_of_week

            try:
                english_tweets.insert_one(tweet)
            except Exception as e:
                print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))





if __name__ == '__main__':

    # connect to MongoDB database
    try:
        client = MongoClient('localhost', 27017) # host, port
    except ConnectionFailure as e:
        sys.stderr.write("Could not connect to MongoDB: %s" % e)
        sys.exit(1)

    # get database with all tweets
    db = client.tweets_database

    print("All collections in the database:")
    print(db.collection_names())

    # get collection containing tweets
    tweets = db.raw_tweets

    # temporary collection with the additional field 'created_at_orig' to be able to
    # calculate the field 'number_of_weeks' in a second step and store it to english_tweets
    temp_tweets = db.temp_tweets

    # create new collection to store the processed english tweets in
    english_tweets = db.english_tweets

    # create temporary collection; field 'created_at_orig' is added
    create_temp_collection(raw_tweets=tweets, temp_tweets=temp_tweets)

    # INFO: In a first step we work only on tweets that are extracted from May 2017
    #       to December 2017
    #       -> set start date to May 1 (ignore older retweets who refer to tweets in the past)
    # REMARK: retweet that refers to oldest original tweet dates from 2008!)
    start_date = datetime.datetime(2017, 5, 1, 0, 00, 00)

    create_english_tweets_collection(temp_tweets, english_tweets, start_date)

    # create index that accelerates operations on the specific field
    temp_tweets.create_index([('created_at_orig', pymongo.ASCENDING)])

    # delete temporary collection
    db.temp_tweets.drop()
