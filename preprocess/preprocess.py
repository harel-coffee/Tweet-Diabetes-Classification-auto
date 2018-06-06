"""

Preprocessing functions

Author: Adrian Ahne
Creation date: 24/04/2018

"""
import string
import unicodedata
import sys
import contractions # expanding contractions
import inflect # natural language related tasks of generating plurals, singular nouns, etc.
import nltk
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


#from emoticons_emoji import preprocess_emot
#from emtion_codes import EMOJI_UNICODE

from emotion_codes import UNICODE_EMOJI
#from emotion_codes import EMOTICONS_UNICODE
from emotion_codes import EMOTICONS
from emotion_codes import EMOJI_TO_CATEGORY
from emotion_codes import Emotions
from defines import *



class Preprocess:

    def __init__(self):
        self.TweetTokenizer = TweetTokenizer()
        # Constant words like URL, USER, EMOT_SMILE, etc. that we want to keep in uppercase
        self.Constant_words = [attr for attr in dir(Constants) if not callable(getattr(Constants, attr)) \
                            and not attr.startswith("__")]+Emotions.EMOTION_CATEGORIES

        self.WN_Lemmatizer = WordNetLemmatizer()

    def get_text(self, raw_tweet):
        """ get text of tweet object in json format """
        return raw_tweet["text"]

    def replace_contractions(self, tweet):
        """ Replace contractions in string of text
            Examples:
              "aren't": "are not",
              "can't": "cannot",
              "'cause": "because",
              "hasn't": "has not",
              "he'll": "he will",

              FIXME: it occurs that
              - "people were in a hurry" is transformed to "we are in a hurry"  !!!
              - "are the main cause of obeisty" -> "are the main because of obesity"
              - "in the U.S are" -> "in the you.S. are"
        """
        return contractions.fix(tweet)

    def replace_hashtags_URL_USER(self, tweet, mode="replace"):
        """
            if mode == "replace"
                Replaces hashtags by its words
                Replaces URLs by the "URL"
                Replace user mentions by "USER"

            if mode == "delete"
                Delete hasthags
                Delete URLs
                Delete USERs

            https://github.com/yogeshg/Twitter-Sentiment

            Ex.:
            s = "@Obama loves #stackoverflow because #people are very #helpful!, \
                 check https://t.co/z2zdz1uYsd"
            print(replace_hashtags_URL_USER(s))
            >> "USER loves stackoverflow because people are very helpful!, check URL"

            TODO: maybe replace @Obama with Obama -> to be checked!
        """

        if mode == "replace":
            # replace URLs
            tweet = Patterns.URL_PATTERN.sub(Constants.URL, tweet)

            # replace mentions : @Obama
            tweet = Patterns.MENTION_PATTERN.sub(Constants.USER, tweet)

            # replace hashtags by its words
            hashtags = Patterns.HASHTAG_PATTERN.findall(tweet)
            for hashtag in hashtags:
                tweet = tweet.replace("#"+hashtag, hashtag)

        elif mode == "delete":
            # replace URLs
            tweet = Patterns.URL_PATTERN.sub("", tweet)

            # replace mentions : @Obama
            tweet = Patterns.MENTION_PATTERN.sub("", tweet)

            # replace hashtags by its words
            hashtags = Patterns.HASHTAG_PATTERN.findall(tweet)
            for hashtag in hashtags:
                tweet = tweet.replace("#"+hashtag, "")

        return tweet

    def tokenize(self, tweet):
        """
            Tokenizes tweet in its single components (words, emojis, emoticons)

            Ex.:
            s = "I love:D python 😄 :-)"
            print(tokenize(s))
            >> ['I', 'love', ':D', 'python', '😄', ':-)']
        """
        return list(self.TweetTokenizer.tokenize(tweet))

    def remove_punctuation(self, tweet):
        """
            Remove punctuations from list of tokenized words

            Punctuations: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~...…

            Example:
            >>> text = ['hallo', '!', 'you', '...', 'going', 'reducing', ',']
            >>> remove_punctuation(text)
            >>> ['hallo', 'you', 'going', 'reducing']

            TODO: check if !,? may contain useful information
        """
        cleaned_tweet = []
        for word in tweet:
            #if word not in string.punctuation and word != '...' and word != '…' and word != '..':
            if word not in string.punctuation and word not in ['...', '…', '..', "\n", "\t", " "] :
                cleaned_tweet.append(word)

        return cleaned_tweet

    def preprocess_emojis(self, tweet):
        '''
            Replace emojis with their emotion category
            Example:
                >>> text = "I love eating 😄"
                >>> preprocess_emoji(text)
                >>> "I love eating EMOT_LAUGH"
        '''

        cleaned_tweet = []
        for ind, char in enumerate(tweet):
            if char in UNICODE_EMOJI:

                if EMOJI_TO_CATEGORY[UNICODE_EMOJI[char]] != "":
                    cleaned_tweet.append(EMOJI_TO_CATEGORY[UNICODE_EMOJI[char]])
                else:
                    print("INFO: No category set for emoji {} -> delete emoji {}".format(char, UNICODE_EMOJI[char]))
            else:
                cleaned_tweet.append(char)

        return cleaned_tweet


    def preprocess_emoticons(self, tweet):
        '''
            Replace emoticons in tweets with their emotion category by searching for
            emoticons with the pattern key word

            Example:
                >>> text = "I like nutella :)"
                >>> preprocess_emoticons(text)
                >>> "I like nutella EMOT_SMILE"
        '''
        cleaned_tweet = []
        for word in tweet:
            match_emoticon = Patterns.EMOTICONS_PATTERN.findall(word)
            if not match_emoticon : # if no emoticon found
                cleaned_tweet.append(word)
            else:
                if match_emoticon[0] is not ':':
                    if match_emoticon[0] is not word:
                        cleaned_tweet.append(word)
                    else:
                        try:
                            cleaned_tweet.append(EMOTICONS[word])
                        except:
                            print("INFO: Could not replace emoticon: {} of the word: {}".format(match_emoticon[0], word), sys.exc_info())
        return cleaned_tweet

    def to_lowercase(self, tweet):
        """
            Convert all characters to lowercase from list of tokenized words

            Example:
                >>> text = ["I", "like", "Nutella", "URL"]
                >>> to_lowercase(text)
                >>> ["i", "like", "nutella", "URL"]

            Remark: Do it after emotion treatment, otherwise smiley :D -> :d
        """
        new_words = []
        for ind,word in enumerate(tweet):
            # if word is not a constant like USER, URL, EMOT_SMILE, etc.
            if word not in self.Constant_words:
                tweet[ind] = word.lower()

        return tweet

    def remove_non_ascii(self, tweet):
        """Remove non-ASCII characters from list of tokenized words"""
        for ind, word in enumerate(tweet):
            # normalize returns the normal fom 'NFKD' of the word
            tweet[ind] = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        return tweet

    def replace_numbers(self, tweet, mode="replace"):
        """
            Replace all interger occurrences in list of tokenized words with textual representation

            Example:
            >>> text = ['June', '2017', 'McDougall', '10', 'Day']
            >>> replace_numbers(text)
            >>> ['June', 'two thousand and seventeen', 'McDougall', 'ten', 'Day']

            REMARK: Maybe better to delete numbers of leave them as string: '2017'
        """

        if mode == "replace":
            p = inflect.engine()
            for ind, word in enumerate(tweet):
                if word.isdigit():
                    tweet[ind] = p.number_to_words(word)

        elif mode == "delete":
            tweet = [word for word in tweet if not word.isdigit() ]

        return tweet

    def remove_stopwords(self, tweet):
        """
            Remove stop words from list of tokenized words

            Example:
            >>> text = ['five', 'reasons', 'to', 'eat', 'like', 'a', 'hunter']
            >>> remove_stopwords(text)
            >>> ['five', 'reasons', 'eat', 'like', 'hunter']
        """
        new_tweet = []
        for word in tweet:
            if word not in Grammar.STOPWORDS: # TODO maybe add manually more stopwords
                new_tweet.append(word)
        return new_tweet

    def lemmatize_verbs(self, tweet):
        """ Lemmatize verbs in list of tokenized words

            Example:
            >>> text = ['americans', 'stopped', 'drinking']
            >>> lemmatize_verbs(text)
            >>> ['americans', 'stop', 'drink']
        """
        for ind, word in enumerate(tweet):
            #tweet[ind] = Grammar.LEMMATIZER.lemmatize(word, pos='v')
            tweet[ind] = self.WN_Lemmatizer.lemmatize(word, pos='v')
        return tweet

    def stem_words(self, tweet, stemmer=Grammar.STEMMER_SNOWBALL):
        """ Stem words in list of tokenized words

            Parameter:
                - tweet :   tokenized list of words of the tweet
                - stemmer : algorithm to use for stemming
                            - Grammar.STEMMER_SNOWBALL (default)
                            - Grammar.PORTER
                            - Grammar.STEMMER_LANCASTER

            Example:
            >>> text = ['predictive', 'tool', 'for', 'children', 'with', 'diabetes']
            >>> stem_words(text)
            >>> ['predict', 'tool', 'for', 'children', 'diabet']

            Remark: Three major stemming algorithms
                - Porter: most commonly used, oldest, most computationally expensive
                - Snowball / Porter2: better than Porter, a bit faster than Porter
                - Lancaster: aggressive algorithm, sometimes to a fault; fastest algo
                            often not intuitiive words; reduces words space hugely
        """
        for ind, word in enumerate(tweet):
            if word not in self.Constant_words: # do not change words like USER, URL, EMOT_SMILE,...
                tweet[ind] = stemmer.stem(word)
        return tweet
