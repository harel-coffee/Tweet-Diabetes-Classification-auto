#!/bin/sh

/space/hadoop/lib/python/bin/python3 runLDA.py \
  --mode "local" \
  --filename "hdfs://bgdta1-demy:8020/data/twitter/wdds/US/personalTweets_mergedAllLocations_US_geoCityCodeNotNull.parquet" \
  --filenameColumns "id, created_at, text, user_screen_name" \
  --saveResultPath "/space/tmp/LDA_results_13032019" \
  --numberTopics "10, 20" 
