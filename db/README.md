# Overview over different files


## TweetsToMongoDB.py
Stores raw tweets, that are extracted via the Twitter API in text files, in the MongoDB database *tweets_database*.

A possible alternative is to extract the tweets via the Twitter API and directly store them in MongoDB. This way is faster and does not need the intermediate step of storing the tweets to text files.


## filter.py
Filters given dataframe after language, with or without retweets and can delete duplicates

This can be done in two modes:
- local mode (-m "local"):
  - Load data
    - from .parquet or .csv to pandas DF (-fn "pathToFile.parquet")
      - Flags:
        - Load only specific columns: --localFileColumns -lfc (-lfc "text, user_name")
        - Specifies column name of text data, default "tweetText": --dataColumnName -dcn (-dcn "text")

    - from MongoDB collection, provide flags:
      - localMongoHost -lh
      - localMongoPort -lp
      - localMongoDatabase -ldb
      - localMongoCollection -lc

- cluster mode (-m "cluster"):
  - load parquet file from hdfs, provide flag (-fn "hdfs://machine:8888/pathToFile.parquet"):
    - dataColumnName -dcn : column in the dataframe containing the text data (default="tweetText")


Filter options:
- By language: --lang , default english (--lang "en")
- Keep retweets or exclude them: --withRetweets (-wr False)
- Add original tweets of retweets: --withOriginalTweetOfRetweet (-wo True)

Furthermore it is possible to rename columnNames if the dataframe was treated before
By default, these columnNames for the tweet object are defined:
ColumnNames = {
    "id" : "id",
    "created_at" : "created_at",
    "lang" : "lang",
    "favorite_count" : "favorite_count",
    "favorited" : "favorited",
    "retweeted" : "retweeted",
    "retweet_count" : "retweet_count",
    "text" : "text",
    "posted_date" : "posted_date",
    "posted_month" : "posted_month",
    "user_id" : "user_id",
    "user_name" : "user_name",
    "user_screen_name" : "user_screen_name",
    "user_followers_count" : "user_followers_count",
    "user_friends_count" : "user_friends_count",
    "user_tweets_count" : "user_statuses_count",
    "user_description" : "user_description",
    "user_time_zone" : "user_time_zone",
    "place_country" : "place_country",
    "place_country_code" : "place_country_code",
    "place_place_type" : "place_place_type",
    "place_name" : "place_name",
    "place_full_name" : "place_full_name",
    "tweet_longitude" : "tweet_longitude",
    "tweet_latitude" : "tweet_latitude",
    "retweeted_user_id" : "retweeted_status_user_id",
    "retweeted_user_name" : "retweeted_status_user_name",
    "retweeted_user_screen_name" : "retweeted_status_user_screen_name",
    "retweeted_user_location" : "retweeted_status_user_location",
    "retweeted_user_created_at" : "retweeted_status_user_created_at",
    "retweeted_user_favourites_count" : "retweeted_status_user_favourites_count",
    "retweeted_user_followers_count" : "retweeted_status_user_followers_count",
    "retweeted_user_friends_count" : "retweeted_status_user_friends_count",
    "retweeted_user_tweet_count" : "retweeted_status_user_tweet_count",
    "retweeted_user_description" : "retweeted_status_user_description",
    "retweeted_user_time_zone" : "retweeted_status_user_time_zone",
    "retweeted_place_country" : "retweeted_status_place_country",
    "retweeted_place_name" : "retweeted_status_place_name",
    "retweeted_place_full_name" : "retweeted_status_place_full_name",
    "retweeted_place_country_code" : "retweeted_status_place_country_code",
    "retweeted_place_place_type" : "retweeted_status_place_place_type",
    "retweeted_created_at" : "retweeted_status_created_at",
    "retweeted_tweet_longitude" : "retweeted_tweet_longitude",
    "retweeted_tweet_latitude" : "retweeted_tweet_latitude",
    "retweeted_text" : "retweeted_status_text"
}

Assuming in your dataframe the column containing the tweet text is called "tweetText",
pass -cD '{"text":"tweetText"}'

Sample call:
python filter.py -m "local" -fn  "hdfs://bgdta1-demy:8020/pathToProject"
                 -lfc "id, lang, text, user_screen_name, user_followers_count, user_friends_count,
                       user_location, user_description, user_tweets_count, place_country,
                       place_full_name,tweet_longitude, tweet_latitude, user_id, retweeted_user_id,
                       retweeted_user_screen_name, retweeted_user_followers_count,
                       retweeted_user_friends_count, retweeted_user_tweet_count,
                       retweeted_user_location, retweeted_user_description, retweeted_place_country,
                       retweeted_place_full_name, retweeted_tweet_longitude, retweeted_tweet_latitude,
                       retweeted_text, is_retweet, posted_month"
                  -s "/space/Work/spark/matching-tweets_diabetes_noRetweetsDuplicates.parquet"
                  -wr "False" -wo "True" -cD '{"retweeted_text":"retweeted_text"}'


## english_noRetweets_tweets.py
Creates the collection *english_noRetweet_tweets* which consists takes the tweets of the collection *english_tweets* and excludes all retweets and duplicates/identical tweets by following steps:
- adds all non-retweet tweets to the collection
- for all the retweets, adds their original tweet to the collection
- Removes all duplicates / identical tweets

This gives us a collection with unique tweets, so each tweet occurs only one time
