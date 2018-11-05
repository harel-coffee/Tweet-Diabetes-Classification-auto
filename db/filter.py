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

import argparse
import pymongo
from pymongo import MongoClient
from pprint import pprint
import datetime
import sys
import json
import os.path as op
import pandas as pd
from ..readWrite.readWrite import savePandasDFtoFile

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library, strInput2bool
from mongoDB_utils import connect_to_database

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
load_library(op.join(basename, 'preprocess'))
from defines import ColumnNames as cn
from defines import Patterns



# def create_temp_collection(raw_tweets, temp_tweets):
#     """
#         Add new field 'created_at_orig' which saves in datetime format the date of the tweet
#         For retweets the date of the original post is saved
#
#         Parameters:
#           - raw_tweets: MongoDB collection of raw tweets
#           - temp_tweets: MongoDB collection in which to store the tweets temporary
#     """
#
#     for tweet in raw_tweets.find({'lang': 'en'}):
#         # if no retweet -> save date of tweet
#         if "retweeted_status" not in tweet.keys():
#             tweet['created_at_orig'] = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
#         # if retweet -> save date of original post
#         else:
#             tweet['created_at_orig'] = datetime.datetime.strptime(tweet['retweeted_status']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
#
#         try:
#             temp_tweets.insert_one(tweet)
#         except Exception as e:
#             print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))
#
#
# def create_english_tweets_collection(temp_tweets, new_collection, start_date):
#     """
#     Create database consisting only of english tweets and add additional field
#     'number_of_weeks' to be able to do more precise analyses in a later step
#
#     INFO:
#     sort tweets by date (s.t. we can label them easily with the number of weeks and so to which bin they belong)
#
#     """
#
#     end_of_week = start_date + datetime.timedelta(days=7)
#     number_of_week = 1
#
#     # sort tweets by date and only greater than start_date
#     for tweet in temp_tweets.find({'created_at_orig': {'$gt': start_date}}).sort('created_at_orig', pymongo.ASCENDING):
#
#         if tweet['created_at_orig'] < end_of_week:
#             tweet['number_of_weeks'] = number_of_week
#
#             try:
#                 new_collection.insert_one(tweet)
#             except Exception as e:
#                 print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))
#
#         else:
#             # go to next week if current tweet was not in the current week bin
#             number_of_week = number_of_week + 1
#             end_of_week = end_of_week + datetime.timedelta(days=7)
#             print("number of weeks:", number_of_week, " end of week: ", end_of_week)
#             tweet['number_of_weeks'] = number_of_week
#
#             try:
#                 new_collection.insert_one(tweet)
#             except Exception as e:
#                 print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))

def add_field_to_collection(collection, URL_PATTERN=False, MENTION_PATTERN=False):
    """
        Function adding a new field to each element of the collection
        The new field will contain the tweet text where the url's are replaced
        by the keyword URL
        This is done to recognise duplicates
        Ex.:   "@BBB I like apples https://t.co/y2fI43"
          and  "@AAA I like apples https://t.co/3Kk4fq"
          and  "@AAA I like apples https://t.co/3Kk4fq"

          are the same tweets but not recognised as the same ones
        For this reason the url's are replaced by the keyword URL and the new
        field "text_URL" is added. This yields:
        "USER I like apples URL"
        "USER I like apples URL"
        "USER I like apples URL"
        Now they can be recognised as duplicates.

    """

    for i, tweet in enumerate(collection.find()):
        text = tweet["text"]
        text = Patterns.URL_PATTERN.sub("URL", text)
        text = Patterns.MENTION_PATTERN.sub("USER", text)

        try:
            collection.update_one({'_id':tweet["_id"]}, {"$set": {"text_URL_USER" : text}}, upsert=False)
        except Exception as e:
            print("Could not update tweet {}: '{}' to MongoDB!".format(tweet["_id"], tweet["text"]), str(e))



def get_duplicates_Mongo(collection):
    """
        Get all tweets that are duplicates (identic)
        Returns a dictionary with the lists of id's of all tweets that are
        copies of one same tweet
    """
    duplicates = collection.aggregate( [
        { "$group": {
            # Group by fields to match on (a,b)
            "_id": { "text_URL_USER": "$text_URL_USER" },

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


def delete_duplicates_Mongo(duplicates, database_collection):
    """
        Deletes all duplicates / identic tweets
        Parameters:
          - duplicates : dict of lists of copies / identical tweets
          - database_collection : collection from which duplicates are deleted
    """

    for duplicate in duplicates:
        n = len(duplicate['docs']) # number of copies of the current tweet

        # delete each duplicate
        i=1
        while i < n:

            result = database_collection.delete_one({'_id': ObjectId(duplicate['docs'][i])})
            #print(result.deleted_count)
            i += 1


def filter_language_Mongo(raw_tweets, filtered_tweets, lang='en'):
    """
        Only consider tweets with the given language and save them in
        the collection filtered_tweets
        ----------------------------------------------------------------------

        Parameters:
          - raw_tweets: MongoDB collection of raw tweets
          - filtered_tweets: MongoDB collection in which to store the filtered tweets
          - lang : language to filter
    """

    for tweet in raw_tweets.find({'lang': lang}):

        try:
            filtered_tweets.insert_one(tweet)
        except Exception as e:
            print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))


def filter_Mongo(raw_tweets, filtered_tweets, lang='en', withRetweets=False, withOriginalTweetOfRetweet=True, deleteDuplicates=True):
    """
        Only consider tweets with the given language and save them in
        the collection filtered_tweets
        ----------------------------------------------------------------------

        Parameters:
          - raw_tweets: MongoDB collection of raw tweets
          - filtered_tweets: MongoDB collection in which to store the filtered tweets
          - lang : language to filter
          - withRetweets : - False : exclude all retweets
                           - True: keep all retweets
          - withOriginalTweetOfRetweet : only if 'withRetweets' == False
                                        adds original tweet of retweet
          - deleteDuplicates: - False: keep duplicates
                              - True: delete duplicates
    """

    if withRetweets:
        for tweet in raw_tweets.find({'lang': lang}):
            try:
                filtered_tweets.insert_one(tweet)
            except Exception as e:
                print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))

    # filter out retweets
    else:
        for tweet in raw_tweets.find({"$and" : [{'lang': lang}, {'retweeted_status' : {"$exists":False}}]}):

            try:
                filtered_tweets.insert_one(tweet)
            except Exception as e:
                print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))

        # add original tweet of retweet
        if withOriginalTweetOfRetweet:
            for tweet in raw_tweets.find( { "retweeted_status": { "$exists" : True }} ):
                tweet_orig = tweet['retweeted_status']

                try:
                    filtered_tweets.insert_one(tweet_orig)
                except Exception as e:
                    print("Could not insert tweet: '{}' to MongoDB!".format(tweet), str(e))

    # delete duplicates
    if deleteDuplicates:

        # replace URL's with keyword URL: so
        URL_PATTERN=Patterns.URL_PATTERN #re.compile(r"http\S+")
        MENTION_PATTERN = Patterns.MENTION_PATTERN #re.compile(r"(?:@[\w_]+)")

        add_field_to_collection(filtered_tweets, URL_PATTERN, MENTION_PATTERN)

        # get all duplicate tweets (copies of tweets)
        duplicates = get_duplicates_Mongo(filtered_tweets)

        # delete duplicates
        delete_duplicates_Mongo(duplicates, filtered_tweets)




def filter_dataframe(raw_tweets, filtered_tweets, configDict, language='en', withRetweets=False,
                     withOriginalTweetOfRetweet=True, deleteDuplicates=True):
    """
        Only consider tweets with the given language and save them in
        the collection filtered_tweets
        ----------------------------------------------------------------------

        Parameters:
          - raw_tweets: pandas or dask dataframe with raw tweets
          - filtered_tweets: dataframe in which to store the filtered tweets
          - lang : language to filter
          - withRetweets : - False : delete all retweets
                           - True: Keep all retweets
          - withOriginalTweetOfRetweet : - True: add original tweet of retweet
                                         - False: don't add original tweet of retweet
          - deleteDuplicates : - True : delete duplicates
                               - False : keep duplicates
    """

    lang = getTweetColumnName("lang", configDict)

    if withRetweets:
        tweets = raw_tweets.loc[raw_tweets[lang] == language] # filter by language
        print("len(withRetweets):", len(tweets))

    # filter out retweets
    else:
        retweeted_text = getTweetColumnName("retweeted_text", configDict)
        print("INFO: Get non-retweets..")
        tweets = raw_tweets.loc[(raw_tweets[lang].values == language) & (raw_tweets[retweeted_text].values == None)] # filter by language and retweet
        print("len(noRetweets):", len(tweets))

        # add original tweets of retweets
        if withOriginalTweetOfRetweet:
            print("INFO: Get retweets..")
            retweets = raw_tweets.loc[(raw_tweets[lang].values == language) & (raw_tweets[retweeted_text].values != None)]
            print("len(Retweets):", len(retweets))

            print("INFO: Add original tweets of retweets..")
            tweets = addOriginalTweetsOfRetweets_df(retweets, tweets)
            print("len(notweets+originalRetweets):", len(tweets))


    if deleteDuplicates:
        # add temporary column with url's and user mentions replaced by constants
        # to better detect duplicates
        print("INFO: Delete duplicates..")
        tweets["tweet_URL_USER"] = [Patterns.MENTION_PATTERN.sub("USER", Patterns.URL_PATTERN.sub("URL", text)) for text in tweets["text"]]

        # delete duplicates
        tweets.drop_duplicates(subset=["tweet_URL_USER"], keep='first', inplace=True)

        # delete temporary created column
        tweets.drop("tweet_URL_USER", axis=1)
        print("len(noduplicates):", len(tweets))

    return tweets



def addOriginalTweetsOfRetweets_df(retweets, tweets):
    """
        Add for each retweet its original tweet (ignore retweets) and delete
        duplicates if existent
        Add dataframe (df) of orignial 'retweets' to existing df of no retweets

        Parameters
        -------------------------------------------------------------
        retweets :  Dataframe with retweets
        tweets :    Dataframe with noRetweets

        Return
        ---------------------------------------------------------------------
        Dataframe with noRetweets and the original tweets of retweets
    """

    # get original tweets from the retweets
    # Remark: The column 'retweet_count' contains -1, to be able to refer to the original tweets later
    originalTweets = pd.DataFrame({"lang" : retweets.lang.values,
                                "user_id" : retweets.retweeted_user_id.values ,
                                "user_name" : retweets.retweeted_user_name.values ,
                                "user_screen_name" : retweets.retweeted_user_screen_name.values ,
                                "user_location" : retweets.retweeted_user_location.values ,
                                "user_created_at" : retweets.retweeted_user_created_at.values ,
                                "user_favourites_count" : retweets.retweeted_user_favourites_count.values ,
                                "user_followers_count" : retweets.retweeted_user_followers_count.values,
                                "user_friends_count" : retweets.retweeted_user_friends_count.values ,
                                "user_tweets_count" : retweets.retweeted_user_tweet_count.values ,
                                "user_description" : retweets.retweeted_user_description.values ,
                                "user_time_zone" : retweets.retweeted_user_time_zone.values ,
                                "place_country" : retweets.retweeted_place_country.values ,
                                "place_name" : retweets.retweeted_place_name.values ,
                                "place_full_name" : retweets.retweeted_place_full_name.values ,
                                "place_country_code" : retweets.retweeted_place_country_code.values ,
                                "place_type" : retweets.retweeted_place_place_type.values ,
                                "tweet_longitude" : retweets.retweeted_tweet_longitude.values ,
                                "tweet_latitude" : retweets.retweeted_tweet_latitude.values ,
                                "text" : retweets.retweeted_text.values,
                                "retweet_count" : [-1 for i in range(len(retweets.lang.values))]
                               }, columns=retweets.columns)

    print("Lenght original tweets:", len(originalTweets))

    # delete duplicates: when a tweet was retweeted several times, keep only one original tweet
    originalTweets.drop_duplicates(subset=["text","user_screen_name"], keep='first', inplace=True)
    print("Lenght original tweets without duplicates:", len(originalTweets))

    # add original tweets (of the retweets) to the non-retweets
    return tweets.append(originalTweets)


def getTweetColumnName(columnName, configDict):
    """
        Define the column name
        Check if columnName is in dict configDict and overwrite the value
        by the one provided from configDict, otherwise take default value
        as defined in class columnNames cn
    """
    if columnName in configDict.keys():
        return configDict[columnName]
    else:
        #return cn.columnName
        return cn[columnName]





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter tweets by language ",
                                     epilog='Example usage in local mode : \
                                             python filter_language.py -m "local" --lang "en" \
                                             --configDict {"lang":"language", "retweeted_text": "retweeted_status.text"} \
                                            ')
    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("--lang", help="Language to filter the tweets", default="en", choices=["en", "fr", "es"])
    parser.add_argument("-lh", "--localMongoHost", help="Host to connect to MongoDB (default=localhost)", default="localhost")
    parser.add_argument("-lp", "--localMongoPort", help="Port to connect to MongoDB (default=27017)", default="27017")
    parser.add_argument("-ldb", "--localMongoDatabase", help="MongoDB database to connect to")
    parser.add_argument("-lc", "--localMongoCollection", help="MongoDB collection (table) in which data is stored")
    parser.add_argument("-lcr", "--localMongoCollectionResult", help="New name of MongoDB collection in which filtered data is stored")
    parser.add_argument("-lpar", "--localParquetfile", help="Path to the parquet file")
    parser.add_argument("-lcsv", "--localCSV", help="Path to the csv file")
    parser.add_argument("-lcsvS", "--localCSVDelimiter", help="Delimiter used in csv file (default=',')", default=",")
    parser.add_argument("-cp", "--clusterPathData", help="Path to the data in cluster mode")
#    parser.add_argument("-dcn", "--dataColumnName", help="If data stored in tabular form, gives the column of the desired text data (default='tweetText')", default="tweetText")
    parser.add_argument("-s", "--saveResultPath", help="Path name where result should be stored")
    parser.add_argument("-wr", "--withRetweets", help="Keep retweets or filter out", choices=(True,False), default=False, type=strInput2bool, nargs='?', const=True)
    parser.add_argument("-wo", "--withOriginalTweetOfRetweet", help="Add original tweets of retweets", choices=(True,False), default=True, type=strInput2bool, nargs='?', const=True)
    parser.add_argument("-cD", "--configDict", help="Configuration dictionary to specify column names; Provide dict in format 'columnName':'desiredColumnName' \
                                                     Possible choices for columnNames: (id, lang, text, retweeted_user_id, retweeted_user_screen_name \
                                                     retweeted_user_location, retweeted_user_created_at, retweeted_user_favourites_count, retweeted_user_followers_count \
                                                     retweeted_user_friends_count, retweeted_user_tweet_count, retweeted_user_description, retweeted_user_time_zone, \
                                                     retweeted_place_country, retweeted_place_name, retweeted_place_full_name, retweeted_place_country_code, \
                                                     retweeted_place_place_type, retweeted_created_at, retweeted_tweet_longitude, retweeted_tweet_latitude, \
                                                     retweeted_text)", type=json.loads, default={})
    args = parser.parse_args()




    if args.mode == "local":

        # check from which source to read the data

        # check if parquet file
        if args.localParquetfile is not None:
            print("Local mode: Read parquet file..")
            raw_tweets = pd.read_parquet(args.localParquetfile, engine="pyarrow")

            filtered_df = filter_dataframe(raw_tweets, args.saveResultPath, args.configDict, language=args.lang,
                            withRetweets=args.withRetweets, withOriginalTweetOfRetweet=args.withOriginalTweetOfRetweet,
                            deleteDuplicates=True)

            saveFile(filtered_df, args.saveResultPath)

        # check if csv
        elif args.localCSV is not None:
            print("Local mode: Read csv file..")
            raw_tweets = pd.read_csv(args.localCSV, sep=args.localCSVDelimiter)

            filtered_df = filter_dataframe(raw_tweets, args.saveResultPath, args.configDict, language=args.lang,
                            withRetweets=args.withRetweets, withOriginalTweetOfRetweet=args.withOriginalTweetOfRetweet,
                            deleteDuplicates=True)

            saveFile(filtered_df, args.saveResultPath)

        # Check if necessary arguments are given
        elif args.localMongoDatabase is None and args.localMongoCollection is None:
            sys.stderr.write("ERROR: A MongoDB database and collection need to be provided to extract the data")
            sys.exit(1)

        # use MongoDB storage
        else:
            print("Local mode: Connect to MongoDB collection..")
            client = connect_to_database()
            db = client[args.localMongoDatabase]
            raw_tweets = db[args.localMongoCollection]
            filtered_tweets = db[args.localMongoCollectionResult]

            # filter language
            filter_language_Mongo(raw_tweets, filtered_tweets, lang=args.lang )



    elif args.mode == "cluster":

        # Check if necessary arguments are given
        if args.clusterPathData is None:
            sys.stderr.write("ERROR: A path to file containing the data needs to be provided")
            sys.exit(1)

        print("Cluster mode: Read parquet files..")
        # TODO: Load data from cluster and tokenize
        raw_tweets = pd.read_parquet(args.clusterPathData, engine="pyarrow")

        filtered_df = filter_dataframe(raw_tweets, args.saveResultPath, args.configDict, language=args.lang,
                                    withRetweets=args.withRetweets, withOriginalTweetOfRetweet=args.withOriginalTweetOfRetweet,
                                    deleteDuplicates=True)

        savePandasDFtoFile(filtered_df, args.saveResultPath)
