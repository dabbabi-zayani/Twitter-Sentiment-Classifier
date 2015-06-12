# this file contains elemantary functions to map a tweet into a vector
# e.g : emoticons score, POS tags counts ...
from __future__ import division
import re

def scoreUnigram(tweet,posuni,neguni,neuuni):
    pos=0
    neg=0
    neu=0
    l=len(tweet.split())

    for w in tweet.split():
        if w in posuni:
            pos+=1
        if w in neguni:
            neg+=1
        if w in neuuni:
            neu+=1
    if (l!=0) :
        pos=pos/l
        neg=neg/l
        neu=neu/l
    return [pos,neg,neu]

# emoticons scores by category , change weighting if needed
def createEmoticonDictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative':-0.5,'Extremely-Negative': -1.0,'Neutral': 0.0}
    emo_score_list={}
    fi = open(filename,"r")
    l=fi.readline()

    while l:
        l=l.replace("\xc2\xa0"," ")
        li=l.split(" ")
        l2=li[:-1]
        l2.append(li[len(li)-1].split("\t")[0])
        sentiment=li[len(li)-1].split("\t")[1][:-1]
        score=emo_scores[sentiment]
        l2.append(score)
        for i in range(0,len(l2)-1):
            emo_score_list[l2[i]]=l2[len(l2)-1]
        l=fi.readline()
    return emo_score_list

def emoticonScore(tweet,d): # d for the emoticons dictionary
    "calculate the aggregate score of emoticons in a tweet"
    s=0.0;
    l=tweet.split(" ")
    nbr=0;
    for i in range(0,len(l)):
        if l[i] in d.keys():
            nbr=nbr+1
            s=s+d[l[i]]
    if (nbr!=0):
        s=s/nbr
    return s

def lenTweet(tweet):
    return len(tweet)

def upperCase(tweet): # returns 1 if there is uppercase words in tweet, 0 otherwise
    result=0
    for w in tweet.split():
        if w.isupper():
            result=1
    return result

def exclamationTest(tweet):
    result=0
    if ("!" in tweet):
        result=1
    return result

def exclamationCount(tweet):
    return tweet.count("!")

def questionTest(tweet):
    result=0
    if ("?" in tweet):
        result=1
    return result

def questionCount(tweet):
    return tweet.count("?")

def freqCapital(tweet): # ratio of number of capitalized letters to the length of tweet
    count=0
    for c in tweet:
        if (str(c).isupper()):
            count=count+1
    if len(tweet)==0:
        return 0
    else:
        
        return count/len(tweet)
# 
# return true if a word is hashtagged
def hashTest(word):
    return word[0]=='#'

# returns list of hashtagged words in a tweet
def hashtagWords(tweet):
    l=tweet.split()
    result=[]
    for w in tweet.split():
        if w[0]=='#' :
            result.append(w)

    return result
    
    

#t=raw_input("tweet :")
#print emoticonScore(t)
