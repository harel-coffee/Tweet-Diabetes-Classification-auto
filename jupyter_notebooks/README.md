# Overview over different jupyter notebooks

## Hub and authority scores.ipynb
Following the paper *The 'who' and 'what' of #diabetes on Twitter* of Mariano Beguerisse-Diaz, Amy K. McLennan, Guillermo Garduno-Hernandez, Mauricio Barahona
and Stanley J. Ulijaszek, hub and authority scores are calculated by for each user to determine the relative importance and influence of a user in their retweet network. 

## TweetsPreprocessing.ipynb
Experimental development of the preprocessing steps for the tweets. Includes:
- replace contractions (can't -> can not)
- replace hashtags (#diabetes -> diabetes), url's (https://protonmail.com/ -> URL), Users (@McKennan -> USER)
- tokenize ("I like diabetes research" -> "I", "like", "diabetes", "research")
- remove punctuations (ex. ;:,?"!)
- categorise emojis 😄 and emoticons :-) into categories like EMOT_SMILE
- all characters to lowercase 
- replace numbers by its alphabetic writing ( 9 -> nine ) 
- remove stopwords (ex.: and, with, a, the)
- lemmatization (ex.: played -> play)
- stemming (ex.: reduce -> reduc) 