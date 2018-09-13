"""
Definitions of basic constants

Author: Adrian Ahne
Creation date: 23/04/2018

Inspired by https://github.com/s/preprocessor/tree/master/preprocessor

- 24/08/2018 : Added Emotion terms
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
from stopword_def import *
from emotionKeywords import *
from emotion_codes import *


class Constants:
    URL = "URL"
    USER = "USER"
    TYPE1 = "TYPEONE"
    TYPE2 = "TYPETWO"

class Patterns:
    URL_PATTERN=re.compile(r"http\S+") # or re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    MENTION_PATTERN = re.compile(r"(?:@[\w_]+)")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)') #TODO check for this

    EMOTICONS_PATTERN = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_UNICODE) + u')', re.IGNORECASE)
    # TODO create EMOJI PATTERN



class Grammar:
    STOPWORDS = stopwords.words('english')
    STOPWORDS_NO_PERSONAL = stopwords_no_personal_list # excludes personal words like "I", "me", "my" to keep them when filtering personal from institutional tweets
    WHITELIST_EN = ["n't", "not", "no", "nor", "never", "nothing", "nowhere", "noone", "none"]
    STEMMER_LANCASTER = LancasterStemmer() # aggressive, fast, sometimes confusing
    STEMMER_PORTER = PorterStemmer(mode='NLTK_EXTENSIONS') # mode that includes further improvements
    STEMMER_SNOWBALL = SnowballStemmer('english') # improved porter

    LEMMATIZER = WordNetLemmatizer()


class WordLists:
    type1 = ["#type1diabetes", "type one", "#type1", "type1", "Type1", "type 1", "T1", "t1",
             "Type 1", "t1d", "T1D", "#t1d", "#T1D", "typeI", "TypeI", "type I", "Type I"]
    type2 = ["#type2diabetes", "type two", "#type2", "type2", "Type2", "type 2", "T2", "t2",
             "Type 2", "t2d", "T2D", "#t2d", "#T2D", "typeII", "TypeII", "type II", "Type II"]

    TYPE1_WORDS = re.compile(u"|".join(type1))
    TYPE2_WORDS = re.compile(u"|".join(type2))

    excludeTweets = ["#chloegracemoretz",  "#chloegmoretz",  "#chloegrace",  "#chloemoretz",
                     "#moretzgrace", "#chloemoretz"] # list with words whose tweets are to be excluded


class Emotions:

    excludeSynonyms_ = ["go", "like", "get", "pull", "floor",  "know", "give",
                        "longer", "please", "handle", "dear", "deal", "substitute",
                        "crank"]
    addWords_ = ["ashamed"]

    parrotsEmotions = get_parrotsEmotions_list_all()
    allEmotions = get_parrotsEmotions_list_all()+DDS_list+PAID_list
    allEmotions_allForms_synonyms = get_synonyms(all_word_forms(allEmotions),
                                                excludeSynonyms_, addWords_)
    # list with all emotions and their synonyms (deleted some which were clearly not an emotion)
    emotions_full_list = emotion_key_words_fulllist

    def get_emotion_synonyms(emotion, excludeWords=excludeSynonyms_, addWords=addWords_,
                             all_emotions_list=emotions_full_list):
        return list(set(get_synonyms(all_word_forms(get_parrotsEmotions_for_emtion(emotion)),
                                                        excludeWords, addWords))
                                     & set(all_emotions_list))




    emotions_synonyms_joy = get_emotion_synonyms("joy")
    emotions_synonyms_love = get_emotion_synonyms("love")
    emotions_synonyms_surprise = get_emotion_synonyms("surprise")
    emotions_synonyms_anger = get_emotion_synonyms("anger")
    emotions_synonyms_sadness = get_emotion_synonyms("sadness")
    emotions_synonyms_fear = get_emotion_synonyms("fear")

    EMOTION_CATEGORIES = Emotions_positive.CATEGORIES_POSITIVE+Emotions_negative.CATEGORES_NEGATIVE
