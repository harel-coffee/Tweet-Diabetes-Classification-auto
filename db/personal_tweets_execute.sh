#!/bin/sh

/space/hadoop/lib/python/bin/python3 personal_tweets.py \
  --mode "local" \
  --pathUserClassifier "/space/Work/spark/classifiers/best_model_classif_user_SVC_2019-01-07_16-08-02.sav" \
  --pathTweetClassifier "/space/Work/spark/classifiers/best_model_classif_tweets_SVC_2019-01-07_17-50-07.sav" \
  --pathWordEmbedding "/space/Work/spark/FastText_model/ft_wordembeddings_09112018.model" \
  --scorePersonalMinimum 0.25 \
  --pathSave "/space/Work/spark/matching-tweets_diab_noRetweetsDupl_personal.parquet" \
  --columnNameTextData "text" \
  --pathData "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/matching-tweets_diabetes_noRetweetsDuplicates.parquet" \











