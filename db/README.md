# Overview over different files


## TweetsToMongoDB.py
Stores raw tweets, that are extracted via the Twitter API in text files, in the MongoDB database *tweets_database*.

A possible alternative is to extract the tweets via the Twitter API and directly store them in MongoDB. This way is faster and does not need the intermediate step of storing the tweets to text files.


## get_unique_twitter_user.py
Creates the collection *unique_users* of the tweets database *tweets_database* in which we store only the unique twitter users who tweet. 


## english_tweets.py
Creates the collection *english_tweets*, which consists only of english tweets (measured by key 'lang' in raw tweet) and following two fields are added to each tweet document:
-  *created_at_orig* : if tweet-document is no retweet -> insert date of the field 'created_at'
                       if tweet-document is retweet -> insert date of original tweet
                                                       of the field 'retweeted_status.created_at'
- *number_of_weeks* : Insert the number of week (int) the tweet is posted based on *created_at_orig*
                      Start date is 01-05-2017 00:00:00

The field *number_of_weeks* will allow more precise analyses in a later step.


## english_noRetweets_tweets.py 
Creates the collection *english_noRetweet_tweets* which consists takes the tweets of the collection *english_tweets* and excludes all retweets and duplicates/identical tweets by following steps:
- adds all non-retweet tweets to the collection
- for all the retweets, adds their original tweet to the collection
- Removes all duplicates / identical tweets 

This gives us a collection with unique tweets, so each tweet occurs only one time