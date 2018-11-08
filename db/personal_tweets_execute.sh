#!/bin/sh

/space/hadoop/lib/python/bin/python3 personal_tweets.py
  --mode "local"
  --pathUserClassifier "/space/Work/spark/classifiers/best_model_user_classif_SVC_2018-08-16_15-47-50.sav"
  --pathTweetClassifier "/space/Work/spark/classifiers/best_model_tweets_classif_SVC_2018-08-16_15-47-50.sav"
  --pathWordEmbedding "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/FastText_model/ft_wordembeddings_08112018.model"
  --pathData "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/matching-tweets_diabetes_noRetweetsDuplicates.parquet"
  --scorePersonalMinimum 0.25
  --pathSave "/space/Work/spark/matching-tweets_diab_noRetweetsDupl_personal.parquet"
  --columnNameTextData "text"
