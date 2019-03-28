"""
Extraction of tweets related to diabetes distress keywords

Author: Adrian Ahne
Creation date: 09/04/2018

Store tweets to file via this command:
>> python tweetsExtraction.py > myfile.txt


Changelog:

20-10-2018 AA
Added command line usage

"""

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time

import argparse
import sys
import os
import os.path as op

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library

load_library(op.join(basename, "readWrite"))

from readWrite import *



#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extracts tweets via Twitter API based on the given list of keywords",
                                     epilog='Example usage: \
                                             python tweetsExtraction.py -at "your_access_token" \
                                                                        -ats "your_access_token_secret" \
                                                                        -ck "your_consumer_key" \
                                                                        -cs "your_consumer_secret" \
                                                                        -pkw "your_path_to_the_file_with_keywords" \
                                            ')
    parser.add_argument("-at", "--access_token", help="Access token for Twitter API", required=True)
    parser.add_argument("-ats", "--access_token_secret", help="Access token secret for Twitter API", required=True)
    parser.add_argument("-ck", "--consumer_key", help="Consumer key for Twitter API", required=True)
    parser.add_argument("-cs", "--consumer_secret", help="Consumer secret for Twitter API", required=True)
    parser.add_argument("-pkw", "--pathKeywords", help="Path to the file containing the keywords to extract the tweets")
    args = parser.parse_args()


#    #Variables that contains the user credentials to access Twitter API
    access_token = args.access_token
    access_token_secret = args.access_token_secret
    consumer_key = args.consumer_key
    consumer_secret = args.consumer_secret
    pathKeywords = args.pathKeywords

    keywords = readToList(pathKeywords)
    print(keywords)


    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    time.sleep(2)

    # filter twitter Streams with the keywords:
    stream.filter(track=keywords)
