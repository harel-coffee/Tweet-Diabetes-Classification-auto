"""
    Remove very similar tweets of a the same user based on cosinus similarity

    For example the cosinus similarity between the two following tweets is over 0.99 
    1:  Fellow Diabetics; I have been on insulin for 9 years now any my A1Cs have always been in the 5 range. It was 6.2 a… https://t.co/b5szHzr5Z8
    2:  Fellow Diabetics; I have been on insulin for 9 years now and my A1Cs have always been in the 5 range. It was 6.2 a… https://t.co/nJ1nGS53s6

    In this way chatbots, who are often tweeting similar content, are identified
"""
import pandas as pd
import numpy as np
from numpy.linalg import norm
import os
import os.path as op
import sys
import itertools
from gensim.models import FastText
from gensim.matutils import softcossim
import argparse

# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library
load_library(op.join(basename, 'preprocess'))
load_library(op.join(basename, 'tweet_utils'))
load_library(op.join(basename, 'readWrite'))
os.environ["HADOOP_HOME"] = "/space/hadoop/hadoop_home"


from tweet_utils import *
from readWrite import readFile, savePandasDFtoFile
from preprocess import Preprocess
prep = Preprocess()


def cosinus_similarity(a, b):
    return np.inner(a,b)/(norm(a)*norm(b))
    #return np.dot(a, b.T)/(norm(a)*norm(b))


def delete_similar_tweets(df):
    if df.shape[0] == 1:
        return df
    else:
        print("shape :", df.shape)
        all_indices = df.index.values.tolist()
        all_combinations = itertools.combinations(all_indices, 2)
        new_indices = []

        while(len(all_indices) > 1):
            first = all_indices[0]
            rest = all_indices[1::]
     
            vec1 = tweet_vectorizer(prep.tokenize(df.loc[first]["text"]), model_ft)#.reshape(1,-1)
            for i in rest:
                vec2 = tweet_vectorizer(prep.tokenize(df.loc[i]["text"]), model_ft)#.reshape(1,-1)

                cos = cosinus_similarity(vec1, vec2)
           
                if (cos > 0.98):
                    print("1: ", df.loc[first]["text"])
                    print("2: ", df.loc[i]["text"])
                    print(cos)
                    print("Remove", i, " :", df.loc[i]["text"])
                    all_indices.remove(i)
            print("append", first, " : ", df.loc[first]["text"])
            new_indices.append(first)
            all_indices.remove(first)

        else:
            if len(all_indices) > 0:
                print("Append last", all_indices[0], " : ", df.loc[all_indices[0]]["text"])
                new_indices.append(all_indices[0])
 
        print("\n New dataframe", df.ix[new_indices].shape)
        print(df.ix[new_indices].head())
        print("\n ----------------------- \n")
        return df.ix[new_indices]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Delete too similar tweets of the same user based on cosinus similarity",
                                    epilog='Example usage in local mode: ')

    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-fn", "--filename", help="Path to the data file", required=True)
    parser.add_argument("-fnd", "--filenameDelimiter", help="Delimiter used in file (default=',')", default=",")
    parser.add_argument("-fnc", "--filenameColumns", help="String with column names to read")
    parser.add_argument("-wep", "--wordembeddingsPath", help="Path to the word embeddings stored in gensim format", required=True)
    parser.add_argument("-gb", "--groupByName", help= "Name of the groupBy column (Default: 'user_name')", default="user_name")
    parser.add_argument("-s", "--saveResultPath", help="Path name where result should be stored", required=True)
    parser.add_argument("-cs", "--similarity", help= "Minimum cosinus similarity from when two tweets are considered equal  (Default: 0.98)", default=0.98)


    args = parser.parse_args()

    if args.mode == "local":
        print("Load file..")
        data = readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter)
        print("Load word embeddings..")
        model_ft = FastText.load(args.wordembeddingsPath)
        
        print("Calculate similarities..")
        newData  = data.groupby(by=args.groupByName).apply(delete_similar_tweets)
        print("Tweets before:", data.shape)
#        newData  = data[data["geo_country_code"] =="US"].groupby(by=args.groupByName).apply(delete_similar_tweets)
#        print("Tweets before:", data[data["geo_country_code"] == "US"].shape)
        print("Tweets after:", newData.shape)

        print("Save tweets without duplicates to {}..".format(args.saveResultPath))
        savePandasDFtoFile(newData, args.saveResultPath)
        

    elif args.mode == "cluster":
        print("ERROR: Not implemented yet!")




