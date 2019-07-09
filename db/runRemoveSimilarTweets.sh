#!/bin/sh

/space/hadoop/lib/python/bin/python3 removeSimilarTweets.py \
        --mode "local" \
        --filename "/space/tmp/matching-tweets_diab_noRT-noDupl_personal_noJokes.parquet" \
        --wordembeddingsPath "/space/tmp/FastText_embedding_20190703/ft_wordembeddings_dim300_minCount5_URL-User-toConstant_iter10_20190703" \
        --groupByName "user_name" \
	--saveResultPath "/space/tmp/matching-tweets_diab_noRT-noBots_personal_noJokes.parquet"


#	--filenameColumns 'id, created_at, lang, retweet_count, text, user_name, user_screen_name, user_location, user_followers_count, user_friends_count, user_tweets_count, user_description, place_full_name, user_id, posted_month, geo_id, geo_name, geo_code, geo_type, geo_country_code, geo_city_code, geo_adm1_code, geo_adm2_code, _score_, _startIndex_, _endIndex_' \
#        --saveResultPath "hdfs:///tmp/personalTweets_placeFullName_noDuplicates_all.parquet" 
