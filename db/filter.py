"""

Author: Adrian Ahne
Date: 27-06-2018

MONGO DB Only
Creates a collection consisting only of english tweets
Furthermore two fields are added to each tweet-document:
  - 'created_at_orig' : if tweet-document is no retweet -> insert date of the field 'created_at'
                        if tweet-document is retweet -> insert date of original tweet
                                                        of the field 'retweeted_status.created_at'

  - 'number_of_weeks' : Insert the number of week (int) the tweet is posted
                        based on 'created_at_orig'
                        Start date is 01-05-2017 00:00:00

The field 'number_of_weeks' will allow more precise analysis in a later step

Last modif: 2019-07-02
Added bot detection and deletion of too similar tweets in the pandas case.

"""

import argparse
from pprint import pprint
import datetime
import sys
import json
import numpy as np
from numpy.linalg import norm
import os
import os.path as op
import itertools
from gensim.models import FastText
import pandas as pd

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library, strInput2bool
#from mongoDB_utils import connect_to_database

load_library(op.join(basename, 'preprocess'))
load_library(op.join(basename, 'readWrite'))
load_library(op.join(basename, 'tweet_utils'))
os.environ["HADOOP_HOME"] = "/space/hadoop/hadoop_home"

from defines import ColumnNames as cn
from defines import Patterns
from readWrite import savePandasDFtoFile, readFile
from tweet_utils import *
from preprocess import Preprocess
prep = Preprocess()


# ----------------------------


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




def cosinus_similarity(a, b):
    return np.inner(a,b)/(norm(a)*norm(b))
    #return np.dot(a, b.T)/(norm(a)*norm(b))


def delete_similar_tweets(df, model_ft, textCol):
    if df.shape[0] == 1:
        return df
    else:
        #print("shape :", df.shape)
        all_indices = df.index.values.tolist()
        all_combinations = itertools.combinations(all_indices, 2)
        new_indices = []

        while(len(all_indices) > 1):
            first = all_indices[0]
            rest = all_indices[1::]

            vec1 = tweet_vectorizer(prep.tokenize(df.loc[first][textCol]), model_ft)#.reshape(1,-1)
            for i in rest:
                vec2 = tweet_vectorizer(prep.tokenize(df.loc[i][textCol]), model_ft)#.reshape(1,-1)

                cos = cosinus_similarity(vec1, vec2)

                if (cos > 0.98):
                    print("1: ", df.loc[first][textCol])
                    print("2: ", df.loc[i][textCol])
                    print(cos)
                    print("Remove", i, " :", df.loc[i][textCol])
                    all_indices.remove(i)
#            print("append", first, " : ", df.loc[first]["text"])
            new_indices.append(first)
            all_indices.remove(first)

        else:
            if len(all_indices) > 0:
 #               print("Append last", all_indices[0], " : ", df.loc[all_indices[0]]["text"])
                new_indices.append(all_indices[0])

        print("\n New dataframe", df.ix[new_indices].shape)
        return df.ix[new_indices]



def filter_dataframe(raw_tweets, configDict, language='en', withRetweets=False,
                     withOriginalTweetOfRetweet=True, deleteDuplicates=True):
    """
        Filters tweets in the provided dataframe
        ----------------------------------------------------------------------

        Parameters:
          - raw_tweets: pandas or dask dataframe with raw tweets
          - lang : language to filter
          - withRetweets : - False : delete all retweets
                           - True: Keep all retweets
          - withOriginalTweetOfRetweet : - True: add original tweet of retweet
                                         - False: don't add original tweet of retweet
          - deleteDuplicates : - True : delete duplicates
                               - False : keep duplicates
    """
    print("Number raw tweets:", len(raw_tweets))

    lang = getTweetColumnName("lang", configDict)
    textCol = getTweetColumnName("text", configDict)

    if withRetweets:
        tweets = raw_tweets.loc[raw_tweets[lang] == language] # filter by language
        print("Number tweets (with retweets) filtered by language {}:".format(lang), len(tweets))

    # filter out retweets
    else:
        retweeted_text = getTweetColumnName("retweeted_text", configDict)

        print("INFO: Get non-retweets..")
        tweets = raw_tweets.loc[(raw_tweets[lang].values == language) & (raw_tweets[retweeted_text].values == None)] # filter by language and retweet
        print("Number tweets (no retweets) filtered by language {}:".format(lang), len(tweets))
        tweets = raw_tweets.loc[(raw_tweets[lang].values == language) & (raw_tweets[retweeted_text].values == None) & (raw_tweets[textCol].values.split(" ")[0] != "RT")] # filter by language and retweet
        print("Number tweets (no retweets, without RT at beginning) filtered by language {}:".format(lang), len(tweets))

        # add original tweets of retweets
        if withOriginalTweetOfRetweet:
            print("INFO: Get retweets..")
            retweets = raw_tweets.loc[(raw_tweets[lang].values == language) & (raw_tweets[retweeted_text].values != None)]
            print("Number retweets :", len(retweets))

            print("INFO: Add original tweets of retweets..")
            tweets = addOriginalTweetsOfRetweets_df(retweets, tweets)
            print("Number of tweets (noRetweets + original tweets of retweets):", len(tweets))


    if deleteDuplicates:
        # 1. Use pandas efficient function drop_duplicates
        #-------------
        # add temporary column with url's and user mentions replaced by constants
        # to better detect duplicates

        print("INFO: Delete duplicates..")
        tweets["tweet_URL_USER"] = [Patterns.MENTION_PATTERN.sub("USER", Patterns.URL_PATTERN.sub("URL", text)) for text in tweets[textCol]]

        # delete duplicates
        tweets.drop_duplicates(subset=["tweet_URL_USER"], keep='first', inplace=True)

        # delete temporary created column
        tweets.drop("tweet_URL_USER", axis=1)
        print("Number tweets without duplicates:", len(tweets))


        # 2. Use word embeddings to delete very close tweets (bots!)
        # --------------------
        print("Load word embeddings..")
        model_ft = FastText.load(args.wordembeddingsPath)

        print("Calculate similarities..")
        tweets = tweets.groupby(by=args.groupByName).apply(lambda group: delete_similar_tweets(group, model_ft, textCol))
        print("Number tweets without bots (very similar tweets):", len(tweets))


            df = readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter)
    print("Filter out tweets about animals..") # likely talking about dog's or cat's diabetes
    animal_list = [" dog", " Dog", " cat ", " Cat ", "cat's"]
    tweets["temp"] = tweets[textCol].map(lambda text: any(animal in text for animal in animal_list))
    tweets = tweets[tweets["temp"] == False]
    del tweets["temp"]
    print("Number tweets without animals:", len(tweets))

    print("----")
    print("Number tweets cleaned total:", len(tweets), "\n")
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

    dict_col = {}
    if "lang" in retweets.columns: dict_col["lang"] = retweets.lang.values
    if "user_id" in retweets.columns: dict_col["user_id"] = retweets.retweeted_user_id.values
    if "user_name" in retweets.columns: dict_col["user_name"] = retweets.retweeted_user_name.values
    if "user_screen_name" in retweets.columns: dict_col["user_screen_name"] = retweets.retweeted_user_screen_name.values
    if "user_location" in retweets.columns: dict_col["user_location"] = retweets.retweeted_user_location.values
    if "user_created_at" in retweets.columns: dict_col["user_created_at"] = retweets.retweeted_user_created_at.values
    if "user_favourites_count" in retweets.columns: dict_col["user_favourites_count"] = retweets.retweeted_user_favourites_count.values
    if "user_followers_count" in retweets.columns: dict_col["user_followers_count"] = retweets.retweeted_user_followers_count.values
    if "user_friends_count" in retweets.columns: dict_col["user_friends_count"] = retweets.retweeted_user_friends_count.values
    if "user_tweets_count" in retweets.columns: dict_col["user_tweets_count"] = retweets.retweeted_user_tweet_count.values
    if "user_description" in retweets.columns: dict_col["user_description"] = retweets.retweeted_user_description.values
    if "user_time_zone" in retweets.columns: dict_col["user_time_zone"] = retweets.retweeted_user_time_zone.values
    if "place_country" in retweets.columns: dict_col["place_country"] = retweets.retweeted_place_country.values
    if "place_name" in retweets.columns: dict_col["place_name"] = retweets.retweeted_place_name.values
    if "place_full_name" in retweets.columns: dict_col["place_full_name"] = retweets.retweeted_place_full_name.values
    if "place_country_code" in retweets.columns: dict_col["place_country_code"] = retweets.retweeted_place_country_code.values
    if "place_type" in retweets.columns: dict_col["place_type"] = retweets.retweeted_place_country_code.values
    if "tweet_longitude" in retweets.columns: dict_col["tweet_longitude"] = retweets.retweeted_tweet_longitude.values
    if "tweet_latitude" in retweets.columns: dict_col["tweet_latitude"] = retweets.retweeted_tweet_latitude.values
    if "text" in retweets.columns: dict_col["text"] = retweets.retweeted_text.values
    if "retweet_count" in retweets.columns: dict_col["retweet_count"] = [-1 for i in range(len(retweets.lang.values))]
    if "is_retweet" in retweets.columns: dict_col["is_retweet"] = [False for i in range(len(retweets.lang.values))]

    originalTweets = pd.DataFrame(dict_col, columns=retweets.columns)

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
        print("Info: For key {} the value {} is defined".format(columnName,configDict[columnName]))
        return configDict[columnName]
    else:
        print("Info: For key {} the value {} is defined".format(columnName,cn[columnName]))
        return cn[columnName]





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter tweets by language ",
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
    parser.add_argument("--lang", help="Language to filter the tweets", default="en", choices=["en", "fr", "es"])
    parser.add_argument("-lh", "--localMongoHost", help="Host to connect to MongoDB (default=localhost)", default="localhost")
    parser.add_argument("-lp", "--localMongoPort", help="Port to connect to MongoDB (default=27017)", default="27017")
    parser.add_argument("-ldb", "--localMongoDatabase", help="MongoDB database to connect to")
    parser.add_argument("-lc", "--localMongoCollection", help="MongoDB collection (table) in which data is stored")
    parser.add_argument("-lcr", "--localMongoCollectionResult", help="New name of MongoDB collection in which filtered data is stored")
    parser.add_argument("-lf", "--filename", help="Path to the data file")
    parser.add_argument("-lfd", "--filenameDelimiter", help="Delimiter used in file (default=',')", default=",")
    parser.add_argument("-lfc", "--filenameColumns", help="String with column names")
    parser.add_argument("-wep", "--wordembeddingsPath", help="Path to the word embeddings stored in gensim format", required=True)
    parser.add_argument("-gb", "--groupByName", help= "Name of the groupBy column (Default: 'user_name')", default="user_name")
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

        if args.filename is not None:
            print("Local mode: Read file..")

            filtered_df = filter_dataframe(readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter),
                                        args.configDict, language=args.lang,
                                        withRetweets=args.withRetweets, withOriginalTweetOfRetweet=args.withOriginalTweetOfRetweet,
                                        deleteDuplicates=True)

            print("Save result to {} ..".format(args.saveResultPath))
            savePandasDFtoFile(filtered_df, args.saveResultPath)



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

        filtered_df = filter_dataframe(raw_tweets, args.configDict, language=args.lang,
                                    withRetweets=args.withRetweets, withOriginalTweetOfRetweet=args.withOriginalTweetOfRetweet,
                                    deleteDuplicates=True)

        print("Save result to {} ..".format(args.saveResultPath))
        savePandasDFtoFile(filtered_df, args.saveResultPath)
