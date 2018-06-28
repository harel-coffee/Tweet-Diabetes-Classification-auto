#ADD PATH TO PYTHONPATH

import sys

# add path for preprocess module to path
sys.path.append('../preprocess')

from preprocess import Preprocess

class Tests:

    def test_preprocessing(tweet="I  can't:D python as @charlie 94 singing in the rain ... #smile :-) https://t.co/gjW9CHMzfH !!"):
        """

        """
        
        prep = Preprocess()
        print("----------ORIGINAL----------:")
        print(tweet)
        print("")

        tweet = prep.replace_contractions(tweet)
        tweet = prep.replace_hashtags_URL_USER(tweet)
        tweet = prep.tokenize(tweet)
        tweet = prep.remove_punctuation(tweet)
        print("-----REPLACED CONTRACTIONS, HASHTAGS, URLs AND USER MENTIONS, TOKENISE; REMOVE PUNCTUATIONS----:")
        print(tweet)
        print("")

        tweet = prep.preprocess_emojis(tweet)
        tweet = prep.preprocess_emoticons(tweet)
        tweet = prep.remove_non_ascii(tweet)
        tweet = prep.to_lowercase(tweet)
        tweet = prep.replace_numbers(tweet)
        print("-----REPLACE EMOJIS AND EMOTICONS; REMOVE NON ASCII, TO LOWERCASE, REPLACE NUMBERS------:")
        print(tweet)
        print("")

        tweet = prep.remove_stopwords(tweet)
        tweet = prep.lemmatize_verbs(tweet)
        tweet = prep.stem_words(tweet)
        print("----------REMOVE STOPWORDS, LEMMATIZE AND STEM-------------:")
        print(tweet)


Tests.test_preprocessing()
