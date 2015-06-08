#!/usr/bin/env python

#import regex
import re

#start process_tweet
def processTweet(tweet,stopWords,slangs): # arg tweet, stopWords list and internet slangs dictionnary
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))','url',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','at_user',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

 #   tweet = tweet.strip('\'"') # removing sepcial caracter
    processedTweet=replaceTwoOrMore(tweet) # replace multi-occurences by two
    words=replaceSlangs(processedTweet,slangs).split()
    processedTweet=''  # result variable
    for w in words:
        #strip punctuation
        if w in stopWords:
            None
        else:
#            w = w.strip('\'"%,.')
#            w=w.replace("'", "")
#            w=w.replace(".", "")
            w=w.replace('''"''', ''' ''')

        #ignore if it is a stop word
            processedTweet=processedTweet+w+' '
    return processedTweet
#end

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

def removeStopWords(tweet,stopWords):
    result=''
    for w in tweet.split(' '):
        if w in stopWords:
            None
        else:
            result=result+w+' '
    return result
def loadSlangs(filename):
    slangs={}
    fi=open(filename,'r')
    line=fi.readline()
    while line:
        l=line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]]=l[1][:-2]
        line=fi.readline()
    fi.close()
    return slangs

def replaceSlangs(tweet,slangs):
    result=''
    words=tweet.split()
    for w in words:
        if w in slangs.keys():
            result=result+slangs[w]+" "
        else:
            result=result+w+" "
    return result



