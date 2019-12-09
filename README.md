# Tweet-Diabetes-Classification

This project aims to identify diabetes distress patterns based on social media data using artificial intelligence methods. We are working with twitter data. 

In the following a small overview over the directories:
- Visualisation_US_map: D3 visualisation of tweets occurrence over the USA after our geolocation algorithm
- Tweets_extraction_Twitter : Extractions of tweets via Twitter API
- WordEmbeddings : Calculating word embeddings (Word2Vec or FastText) via the gensim package
- data : Trained models (not up-to-date)
- db : Algorithms to filter tweets, remove duplicates (from chatbots) , clean database , ..
- files : Only list with keywords to extract tweets
- jupyter_notebooks : To experiment 
- preprocess : Functions to preprocess tweets and textual data in general
- readWrite : Read & Write files (parquet, csv, text)
- tests : (not up-to-date)
- topicModel : Extract topics with LDA method
- training : Train classifiers for filtering or predicting
- utils : utility functions


More detailed information about the programs and algorithms used, you will find in the corresponding folders.

Check the development branch 'devAA' for current programs.

## Prerequisites 
- Python (version >= 3.5.5)

