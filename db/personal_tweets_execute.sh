#!/bin/sh

/space/hadoop/lib/python/bin/python3 personal_tweets.py \
  --mode "local" \
  --pathUserClassifier "/space/tmp/bestModel_personalTweets_20190703.model" \
  --pathTweetClassifier "/space/tmp/bestModel_personalUsers_20190703.model" \
  --pathWordEmbedding "/space/tmp/FastText_embedding_20190703/ft_wordembeddings_dim300_minCount5_URL-User-toConstant_iter10_20190703" \
  --scorePersonalMinimum 0.25 \
  --pathSave "/space/tmp/matching-tweets_diab_noRT-noDupl_personal.parquet" \
  --columnNameTextData "text" \
  --pathData "/space/tmp/matching-tweets_diab_noRT-noDupl.parquet"

#  --pathData "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/matching-tweets_diabetes_noRetweetsDuplicates.parquet" \









