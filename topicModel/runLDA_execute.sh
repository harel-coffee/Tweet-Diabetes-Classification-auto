
python runLDA.py \
  --mode "local" \
  --filename "/Users/Adrian/Desktop/WDDS/Models_Data/tweets_08042019/matching-tweets_diab_noRetweetsDupl_personal_noJokes_LocationUS_geoCityCodeNotNull.parquet" \
  --filenameColumns "id, created_at, text, user_screen_name" \
  --saveResultPath "/Users/Adrian/Desktop/WDDS/TopicModels/LDA_08042019" \
  --numberTopics "30, 50, 100" 
