# Overview over different files


## TweetsToMongoDB.py
Stores raw tweets, that are extracted via the Twitter API in text files, in the MongoDB database $tweets_database$.

A possible alternative is to extract the tweets via the Twitter API and directly store them in MongoDB. This way is faster and does not need the intermediate step of storing the tweets to text files.


## get_unique_twitter_user.py
Creates the collection $unique_users$ of the tweets database $tweets_database$ in which we store only the unique twitter users who tweet.  \textif{coucou} 

