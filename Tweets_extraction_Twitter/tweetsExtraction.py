"""
Extraction of tweets related to diabetes distress keywords

Author: Adrian Ahne
Creation date: 09/04/2018

Store tweets to file via this command:
>> python tweetsExtraction.py > myfile.txt

"""

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time

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


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    time.sleep(2)

    # filter twitter Streams with the keywords:
    stream.filter(track=keywords)
