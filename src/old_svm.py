# the actual classifier script for predicting a sentiment using SVM
from __future__ import division
from sklearn import svm
from sklearn import cross_validation

import numpy as np


import features
import polarity
import ngramGenerator
import preprocessing


KERNEL_FUNCTION='linear'

print "Initializing dictionnaries"
stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')
sentiWordnet=polarity.loadSentiFull('../resources/sentiWordnetBig.csv')
emoticonDict=features.createEmoticonDictionary("../resources/emoticon.txt")

print "Bulding 150 unigram vector"
positive=ngramGenerator.mostFreqList('../data/positive_processed.csv',500)
negative=ngramGenerator.mostFreqList('../data/negative_processed.csv',500)
neutral=ngramGenerator.mostFreqList('../data/neutral_processed.csv',500)


total=positive+negative+neutral # total unigram vector removing cross occurences 
for w in total:
    count=total.count(w)
    if (count > 1):
        while (count>0):
            count=count-1
            total.remove(w)
#print total
 
def mapTweet(tweet,sentiWordnet,emoDict,unigram,slangs):
    out=[]
    line=preprocessing.processTweet(tweet,stopWords,slangs)
   
    p=polarity.polarity(line,sentiWordnet)
   
    out.append(float(p[0])-float(p[1])) # aggregate polarity for pos neg and neutral here neutral is stripped
    pos=polarity.posFreq(line,sentiWordnet)
#    out.extend([float(pos['v']),float(pos['n']),float(pos['a']),float(pos['r'])]) # pos counts inside the tweet
#    out.append(float(features.emoticonScore(line,emoDict))) # emo aggregate score be careful to modify weights
    out.append(float(len(features.hashtagWords(line))/40)) # number of hashtagged words
    out.append(float(len(line)/140)) # for the length
    out.append(float(features.upperCase(line))) # uppercase existence : 0 or 1
#    out.append(float(features.exclamationTest(line)))
    out.append(float(line.count("!")/140))
    out.append(float((features.questionTest(line))))
#    out.append(float(line.count('?')/140))
    out.append(float(features.freqCapital(line)))
    for w in unigram:  # unigram
            if (w in line):
                out.append(float(1))
            else:
                out.append(float(0))
    return out
# load matrix
def loadMatrix(posfilename,neufilename,negfilename,poslabel,neulabel,neglabel):
    vectors=[]
    labels=[]
    f=open(posfilename,'r')
    kpos=0
    kneg=0
    kneu=0
    line=f.readline()
    while line:
        kpos=kpos+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(poslabel))
        line=f.readline()
        print str(kpos)+"positive line loaded"+str(len(vectors))+" "+str(len(labels))
    f.close()
    
    f=open(neufilename,'r')
    line=f.readline()
    while line:
        kneu=kneu+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neulabel))
        line=f.readline()
        print str(kneu)+"neutral lines loaded"
    f.close()
    
    f=open(negfilename,'r')
    line=f.readline()
    while line:
        kneg=kneg+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neglabel))
        line=f.readline()
        print str(kneg)+"negative lines loaded"
    f.close()
    return vectors,labels


# map tweet into a vector 
def trainModel(X,Y,knel):
    clf=svm.SVC(kernel=knel) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf

def predict(tweet,model): # test a tweet against a built model 
    z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs) # mapping
    return model.predict([z]).tolist() # transform nympy array to list 

def predictFile(filename,svm_model): # function to load test file in the csv format : sentiment,tweet 
    f=open(filename,'r')
    fo=open(filename+".result",'w')
    line=f.readline()
    while line:
        tweet=line[:-1]

        nl=predict(tweet,svm_model)
    
        fo.write(r'"'+str(nl)+r'","'+tweet+r'"\n')
        line=f.readline()
   
    f.close()
    fo.close()
    print "Tweets are classified . The result is in "+filename+".result"

def loadTest(filename): # function to load test file in the csv format : sentiment,tweet 
    f=open(filename,'r')
    line=f.readline()
    labels=[]
    vectors=[]
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]

        z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(s)
        line=f.readline()
#        print str(kneg)+"negative lines loaded"
    f.close()
    return vectors,labels

def batchPredict(vectors,model): # the output is a numpy array of labels
    return model.predict(vectors).tolist()

def testModel(vectors,labels,model): # for a given set of labelled vectors calculate model labels and give accuract
    a=0 # wrong classified vectors
    newLabels=model.predict(vectors).tolist()
    for i in range(0,len(newLabels)):
        if newLabels[i]!=labels[i]:
            a=a+1
    if len(labels)==0:
        return 0.0
    else:
        return 1-a/len(labels) # from future import dividion


# loading training data
print "Loading training data"
X,Y=loadMatrix('../data/positive_processed.csv','../data/neutral_processed.csv','../data/negative_processed.csv','4','2','0')
#X,Y=loadMatrix('../data/small_positive_processed.csv','../data/small_neutral_processed.csv','../data/small_negative_processed.csv','4','2','0')

x=np.array(X)
y=np.array(Y)
print "Optimizing model"
KERNEL_FUNCTIONS=['linear']
C=[0.01*i for i in range(1,11)]
ACC=0.0
Iter=0
for knel in KERNEL_FUNCTIONS:
    for c in C:
        Iter=Iter+1
        clf = svm.SVC(kernel=KERNEL_FUNCTION, C=c)
        scores = cross_validation.cross_val_score(clf, x, y, cv=5)
        if (scores.mean() > ACC):
            ACC=scores.mean()
            KERNEL_FUNCTION=knel
            C_PARAMETER=c

        print "Iteration No : "+str(Iter)+" Kernel : "+knel+", c = "+str(c)
        #print scores # the precision for five iterations
        print("Accuracy of the model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 
print "best C : "+str(C_PARAMETER)
print "best knel : "+KERNEL_FUNCTION

# 5 fold cross validation
print "Performing 5 cross fold validation"
clf = svm.SVC(kernel=KERNEL_FUNCTION, C=C_PARAMETER)
scores = cross_validation.cross_val_score(clf, x, y, cv=5)
print scores # the precision for five iterations
print("Accuracy of the model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 
MODEL=trainModel(X,Y,KERNEL_FUNCTION) # poly of degree 3 (default)

 

# uncomment to classify test dataset 
print "Loading test dataset..."
V,L=loadTest('../data/test_dataset.csv')

print "Classifying test dataset..."
print "Classification done : Performance over test dataset : "+str(testModel(V,L,MODEL))

user_input=raw_input("Write a tweet to test or a file path for bulk classification with svm model. press q to quit\n")
while user_input!='q':
    try:
        predictFile(user_input,MODEL)
    except:
        print "sentiment : "+str(predict(user_input,MODEL))
        user_input=raw_input("Write a tweet to test or a file path for bulk classification . press q to quit\n")

# the end !
