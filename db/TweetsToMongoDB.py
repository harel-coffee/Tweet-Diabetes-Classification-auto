"""
Store raw tweets that are stored in txt. files in NoSQL database MongoDB

Author: Adrian Ahne
Creation date: 16/04/2018

"""



from pymongo import MongoClient
import json
import os
import sys

# list of diabetes related key-words
key_words = ['glucose', '#glucose', 'blood glucose', '#bloodglucose', 'insulin', '#insulin',
                        'insulin pump', '#insulinpump', 'diabetes', '#diabetes', 't1d', '#t1d',
                        '#type1diabetes', '#type1', 't2d', '#t2d', '#type2diabetes', '#type2',
                        '#bloodsugar', '#dsma', '#bgnow', '#wearenotwaiting', '#insulin4all', 'dblog',
                        '#dblog', 'diyps', '#diyps', 'hba1c', '#hba1c', '#cgm', '#freestylelibre',
                        'diabetic', '#diabetic', '#gbdoc', 'finger prick', '#fingerprick', '#gestational',
                        'gestational diabetes', '#gdm', 'freestyle libre', '#changingdiabetes',
                        'continuous glucose monitoring', '#continuousglucosemonitoring', '#thisisdiabetes',
                        '#lifewithdiabetes', '#stopdiabetes', '#diabetesadvocate', '#diabadass',
                        '#diabetesawareness', '#diabeticproblems', '#diaversary', '#justdiabeticthings',
                        '#diabetestest', '#t1dlookslikeme', '#t2dlookslikeme']

# connect to Mongo database on localhost
try:
    client = MongoClient('localhost', 27017) # host, port
except ConnectionFailure as e:
    sys.stderr.write("Could not connect to MongoDB: %s" % e)
    sys.exit(1)

# create database tweet_database
db = client.tweets_database

# delete collection
#db.tweets.drop()

# create collection tweets (eq. of a table in relational database)
tweets = db.tweets

# path to twitter data
rootPath =r'D:\A_AHNE1\Twitter_data\space\extracted data\data\twitter\diabetes'

# walk through all directories of rootPath and store each tweet as document in MongoDB
for root, dirs, files in os.walk(rootPath):

    for filename in files:
        filename_absolut = os.path.join(root, filename)

        # read text file containing tweets
        try:
            with open(filename_absolut, 'r', encoding="utf8") as f:
                for line in f:
                    tweet = json.loads(line)

                    try:
                        tweet_text = tweet["text"]

                        # checks if any keyword of the diabetes key word list is in the tweet and
                        # adds the tweet to the database
                        # ! At the beginning of the Twitter extraction the keyword list was different,
                        # ! this is way we filter again
                        if any(keyword in tweet_text for keyword in key_words):
                            tweets.insert_one(tweet)

                    except Exception as e:
                        print(str(e))

                f.close()

        except:
            print("Could not open file: {}".format(filename_absolut), sys.exc_info())
