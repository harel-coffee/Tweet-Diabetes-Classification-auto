"""
Extraction of tweets related to diabetes distress keywords

Author: Adrian Ahne
Creation date: 09/04/2018

Store tweets to file via this command:
>> python tweetsExtraction.py > myfile.txt

<<<<<<< HEAD
=======
Changelog:

20-10-2018 AA
Added command line usage

>>>>>>> devAA
"""

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
<<<<<<< HEAD

#Variables that contains the user credentials to access Twitter API
access_token = "..."
access_token_secret = "..."
consumer_key = "..."
consumer_secret = "..."


keywords = ['glucose', '#glucose', 'blood glucose', '#bloodglucose', 'insulin', '#insulin',
            'insulin pump', '#insulinpump', 'diabetes', '#diabetes', 't1d', '#t1d',
            '#type1diabetes', '#type1', 't2d', '#t2d', '#type2diabetes', '#type2',
            '#bloodsugar', '#dsma', '#bgnow', '#wearenotwaiting', '#insulin4all', 'dblog',
            '#dblog', 'diyps', '#diyps', 'hba1c', '#hba1c', '#cgm', '#freestylelibre',
            'diabetic', '#diabetic', '#gbdoc', 'finger prick', '#fingerprick', '#gestational',
            'gestational diabetes', '#gdm', 'freestyle libre', '#changingdiabetes',
            'continuous glucose monitoring', '#continuousglucosemonitoring', '#thisisdiabetes',
            '#lifewithdiabetes', '#stopdiabetes', '#diabetesadvocate', '#diabadass',
            '#diabetesawareness', '#diabeticproblems', '#diaversary', '#justdiabeticthings',
            '#diabetestest', '#t1dlookslikeme', '#t2dlookslikeme', '#duckfiabetes' ,
            '#kissmyassdiabetes', '#GBDoc',

            'glycémie', 'glycemie', "#glycémie", '#glycemie', 'insuline', '#insuline',
            'pompe à insuline', 'pompe a insuline', '#pompeàinsuline', '#pompeainsuline',
            'diabete', 'diabète', 'DT1', '#DT1', '#diabetetype1', '#diabètetype1',
            '#type1', 'DT2', '#DT2', '#diabetetype2', '#diabètetype2', '#type2',
            'diabète sucre', 'diabete sucre', '#diabètesucré', '#diabètesucre',
            '#diabetesucré', '#diabetesucre', 'diabétique', 'diabetique',
            '#diabétique', '#diabetique', '#changingdiabetes', 'hba1c', '#hba1c',
            'freestyle libre', '#freestylelibre', '#cgm', 'diabète gestationnel',
            '#diabetegestationnel', '#diabètegestationnel',

            'Glucose', 'glukose', 'Glukose', '#Glucose', '#glukose', '#Glukose',
            'Blutzucker', 'blutzucker', 'Glykämie', 'glykämie' , '#Blutzucker',
            '#blutzucker', '#Glykämie', '#Insulin', '#insulin', 'Insulin', 'insulin',
            'Insulinpumpe', 'insulinpumpe', '#Insulinpumpe', '#insulinpumpe', 'diabetes',
            'Diabetes', '#diabetes', '#Diabetes', 'Diabetiker', 'diabetiker',
            '#Diabetiker', '#diabetiker', '#Diabetikern', '#diabetikern', 't1d',
            '#t1d', '#T1diabetes', '#T2D', '#T1D', '#Diabetespatienten',
            '#diabetespatienten', '#Diabetespatient', '#diabetespatient', '#TD1',
            '#TD2', '#dedoc', '#diabetestyp1', '#diabetestyp2', "#DiabetesTyp1",
            "#Diabetestyp1", "#DiabetesTyp2",
            "#Diabetestyp2", "#Glukosekontrolle", "#glukosekontrolle", "#Unterzuckerungen",
            "#unterzuckerungen", "#Unterzuckerung", "#unterzuckerung", "#Hypoglykämie",
            "#hypoglykämie", "#Glukosemonitoring", "#glukosemonitoring", "#Blutzuckerwerte",
            "#blutzuckerwerte", "#menschmitdiabetes", "#Duckfiabetes", "#kindermittyp1",
            "Freestyle libre", "freestyle libre", "#freestylelibre", "#Freestylelire",
            "#FreeStyleLibre", "#FreestyleLibre", "#diabeteskids", "#Diabeteskids",
            "#DiabetesKids", "#Antidiabetika", "#antidiabetika", "Antidiabetika",
            "antidiabetika", "diabetisch", "#diabetisch", "hba1c", "Hba1c", "#hba1c",
            "Hba1c"
            ]
=======
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

>>>>>>> devAA


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


<<<<<<< HEAD
if __name__ == '__main__':

=======

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


>>>>>>> devAA
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    time.sleep(2)

    # filter twitter Streams with the keywords:
    stream.filter(track=keywords)
