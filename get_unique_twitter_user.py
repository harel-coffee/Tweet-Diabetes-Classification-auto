"""
Get names and ids of the persons who are tweeting and are mentioned
and save them in a collection of MongoDB
(Excluding persons who are retweeting)

Example document being saved to MongoDB:

 {
    'id': 801330528,
    'id_str': '801330528',
    'name': 'PenGwen',
    'screen_name': 'PenGwenart'
  }


Author: Adrian Ahne
Creation date: 18/04/2018

"""

from pymongo import MongoClient
from pprint import pprint


try:
    client = MongoClient('localhost', 27017) # host, port
except ConnectionFailure as e:
    sys.stderr.write("Could not connect to MongoDB: %s" % e)
    sys.exit(1)

# get database with all tweets
db = client.tweets_database

# get collection containing tweets (eq. of a table in relational database)
tweets = db.tweets

# create new collection in which unique twitter persons are stored
unique_users = db.unique_users

# compare persons that should be added with this list to avoid that the
# same person is stored several times
already_inserted_persons = []

# only english tweets
for tweet in tweets.find({'lang': 'en'}):

    # consider only persons of non-retweets
    if 'retweeted_status' not in tweet.keys():

        # create dict of user information to be stored
        user = {
                'id' : tweet['user']['id'],
                'id_str' : tweet['user']['id_str'],
                'name' : tweet['user']['name'],
                'screen_name' : tweet['user']['screen_name']
        }

        # add user only if he has not already be stored (uniqueness of each user!)
        if user not in already_inserted_persons:
            try:
                unique_users.insert_one(user)
            except Exception as e:
                print("Could not insert user {} to MongoDB!".format(user), str(e))

            already_inserted_persons.append(user)

        # add users that are mentioned in tweet
        for mentioned_user in tweet['entities']['user_mentions']:

            user = {
                    'id' : mentioned_user['id'],
                    'id_str' : mentioned_user['id_str'],
                    'name' : mentioned_user['name'],
                    'screen_name' : mentioned_user['screen_name']
            }

            # add user only if he has not already be stored (uniqueness of each user!)
            if user not in already_inserted_persons:
                try:
                    unique_users.insert_one(user)
                except Exception as e:
                    print("Could not insert user {} to MongoDB!".format(user), str(e))

                already_inserted_persons.append(user)

#print(unique_users_list)
