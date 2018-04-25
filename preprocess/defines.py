"""
Definitions of basic constants

Author: Adrian Ahne
Creation date: 23/04/2018

Inspired by https://github.com/s/preprocessor/tree/master/preprocessor
"""


import re
import nltk
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# download library
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from emotion_codes import EMOTICONS_UNICODE

class Constants:
    URL = "URL"
    USER = "USER"

class Patterns:
    URL_PATTERN=re.compile(r"http\S+") # or re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    MENTION_PATTERN = re.compile(r"(?:@[\w_]+)")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)') #TODO check for this

    EMOTICONS_PATTERN = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_UNICODE) + u')', re.IGNORECASE)
    # TODO create EMOJI PATTERN


class Grammar:
    STOPWORDS = stopwords.words('english')
    STEMMER_LANCASTER = LancasterStemmer() # aggressive, fast, sometimes confusing
    STEMMER_PORTER = PorterStemmer(mode='NLTK_EXTENSIONS') # mode that includes further improvements
    STEMMER_SNOWBALL = SnowballStemmer('english') # improved porter

    LEMMATIZER = WordNetLemmatizer()

# class Emotions:
#     EMOT_SMILE = "EMOT_SMILE"
#     EMOT_LAUGH = "EMOT_LAUGH"
#     EMOT_LOVE = "EMOT_LOVE"
#     EMOT_WINK = "EMOT_WINK"
#     EMOT_SAD_ANGRY = "EMOT_SAD_ANGRY"
#     EMOT_SURPRISED = "EMOT_SURPRISED"
#     EMOT_CRY = "EMOT_CRY"
#
#
#     EMOTION_CATEGORIES = [EMOT_SMILE, EMOT_LAUGH, EMOT_LOVE, EMOT_WINK, EMOT_SAD_ANGRY,
#                             EMOT_SURPRISED, EMOT_CRY]
