# python script for determining the polarity and POS characteristics
# of an input tweet using SentiWordNet3.0 dictionnary
from __future__ import division
import features
import nltk # for pos tagging 

# load input file in a dictionnary
def loadSentiSimple(filename):
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        tag=l[0]
        word=l[1]
        pos=abs(float(l[3]))
        neg=abs(float(l[4]))
        neu=abs(float(l[5]))

        output[word]=[tag,pos,neg,neu]
        line=fi.readline()
    fi.close()
    return output

def loadSentiFull(filename): # need fixing , use loadSentiSmall instead 
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        try:
            tag=l[0]
            sentence=l[4]
            new = [word for word in sentence.split() if (word[-2] == "#" and word[-1].isdigit())]
            pos=abs(float(l[2]))
            neg=abs(float(l[3]))
            neu=float(1-pos-neg)
        except:
#            print line
            line=fi.readline()
            continue

        for w in new:
            output[w[:-2]]=[tag,pos,neg,neu]
        line=fi.readline()
    fi.close()
    return output

# combine words with their pos tags function
def loadSentiWordnet(filename): # need fixing , use loadSentiFull instead 
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        try:
            tag=l[0]
            sentence=l[4]
            new = [word for word in sentence.split() if (word[-2] == "#" and word[-1].isdigit())]
            pos=abs(float(l[2]))
            neg=abs(float(l[3]))
            neu=float(1-pos-neg)
        except:
#            print line
            line=fi.readline()
            continue

        for w in new:
            output[(w[:-2],tag)]=[pos,neg,neu] # dict(word,tag)=scores
        line=fi.readline()
    fi.close()
    return output


def polarity(tweet,sentDict): # polarity aggregate of a tweet from sentiWordnet dict
    pos=0.0
    neg=0.0
    neu=0.0
    n_words=0
    for w in tweet.split():
        if w in sentDict.keys():
            n_words=n_words+1
            pos=pos+sentDict[w][1]
            neg=neg+sentDict[w][2]
            neu=neu+sentDict[w][3]
        if features.hashTest(w) and w[1:] in sentDict.keys():
            pos=pos+2*sentDict[w[1:]][1] # more weight for hashed words
            neg=neg+2*sentDict[w[1:]][2]
            neu=neu+2*sentDict[w[1:]][3]
            
    if (n_words ==0 ):
        return [pos,neg,neu]
    else:
        return [pos/n_words,neg/n_words,neu/n_words]

# function for polarity combined with pos using output of loadSeniWordnet
def posPolarity(tweet,sentDict): # polarity aggregate of a tweet from sentiWordnet dict
    pos=0.0
    neg=0.0
    neu=0.0
   
    posScores={}
    posNumber={}

    posNumber['a']=0
    posNumber['n']=0
    posNumber['v']=0
    posNumber['r']=0

    posScores['a']=0.0
    posScores['n']=0.0
    posScores['v']=0.0
    posScores['r']=0.0

    n_words=0
    nlpos={}
    nlpos['a']=['JJ','JJR','JJS'] # adjective tags in nltk
    nlpos['n']=['NN','NNS','NNP','NNPS'] # nouns ...
    nlpos['v']=['VB','VBD','VBG','VBN','VBP','VBZ','IN'] # verbs 
    nlpos['r']=['RB','RBR','RBS'] # adverbs 

    text=tweet.split()
    tags=nltk.pos_tag(text)

    for z in tags:
        w=("testtesttestt",'mdr')
        y=list(z)
        if (y[1] in nlpos['a']):
            w=(y[0],'a')
            posNumber['a']+=1
        if (y[1] in nlpos['n']):
            w=(y[0],'n')
            posNumber['n']+=1
        if (y[1] in nlpos['v']):
            w=(y[0],'v')
            posNumber['v']+=1
        if (y[1] in nlpos['r']):
            w=(y[0],'r')
            posNumber['r']+=1



        if w in sentDict.keys():
            n_words=n_words+1
            posScores[w[1]]+=sentDict[w][0]-sentDict[w][1]
            pos=pos+sentDict[w][0]
            neg=neg+sentDict[w][1]
            neu=neu+sentDict[w][2]

        if features.hashTest(list(w)[0]) and (list(w)[0][1:],list(w)[1]) in sentDict.keys():
            n_words=n_words+1
            posNumber[w[1]]+=1
            posScores[w[1]]+=sentDict[(list(w)[0][1:],list(w)[1])][0]-sentDict[(list(w)[0][1:],list(w)[1])][1]
            pos=pos+2*sentDict[(list(w)[0][1:],list(w)[1])][0] # more weight for hashed words
            neg=neg+2*sentDict[(list(w)[0][1:],list(w)[1])][1]
            neu=neu+2*sentDict[(list(w)[0][1:],list(w)[1])][2]
            
    if (n_words ==0 ):
        return [pos,neg,neu,posScores['a'],posScores['n'],posScores['v'],posScores['r'],posNumber['a'],posNumber['n'],posNumber['v'],posNumber['r']]
    else:
        return [pos/n_words,neg/n_words,neu/n_words,posScores['a']/n_words,posScores['n']/n_words,posScores['v']/n_words,posScores['r']/n_words,posNumber['a']/n_words,posNumber['n']/n_words,posNumber['v']/n_words,posNumber['r']/n_words] # do not use the neutral scoore it will fuck up evertg

def posFreq(tweet,dict): # calculates the frequency of apperances of pos in a tweet
    result={}
    result['v']=0
    result['n']=0
    result['a']=0
    result['r']=0
    nbr=0
    for w in tweet.split():
        if (w in dict.keys()):
            nbr=nbr+1
            result[dict[w][0]]=result[dict[w][0]]+1
    if (nbr != 0):
        result['v']=result['v']/nbr
        result['a']=result['a']/nbr
        result['n']=result['n']/nbr
        result['r']=result['r']/nbr
    return result

