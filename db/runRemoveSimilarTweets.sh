#!/bin/sh

/space/hadoop/lib/python/bin/python3 removeSimilarTweets.py \
        --mode "local" \
        --filename "hdfs:///tmp/personalTweets_placeFullName.parquet" \
        --wordembeddingsPath "/space/Work/spark/FastText_model/ft_wordembeddings_09112018.model" \
        --groupByName "user_name" \
	--filenameColumns 'id, created_at, lang, retweet_count, text, user_name, user_screen_name, user_location, user_followers_count, user_friends_count, user_tweets_count, user_description, place_full_name, user_id, posted_month, geo_id, geo_name, geo_code, geo_type, geo_country_code, geo_city_code, geo_adm1_code, geo_adm2_code, _score_, _startIndex_, _endIndex_' \
        --saveResultPath "hdfs:///tmp/personalTweets_placeFullName_noDuplicates_all.parquet" 
