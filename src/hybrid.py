from __future__ import division
from sklearn import neighbors
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing as pr

from sklearn.feature_selection import SelectKBest, f_classif # for features selection


import numpy as np

import features
import polarity
import ngramGenerator
import preprocessing


# User input for model parameters
N_NEIGHBORS=10  # number of neighbors for KNN
KERNEL_FUNCTION='linear'  # kernel function for SVM
C_PARAMETER=0.2
UNIGRAM_SIZE=1000



print "Initializing dictionnaries"
stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')
#sentiWordnet=polarity.loadSentiFull('../resources/sentiWordnetBig.csv')
sentiWordnet=polarity.loadSentiWordnet('../resources/sentiWordnetBig.csv')

emoticonDict=features.createEmoticonDictionary("../resources/emoticon.txt")

print "Bulding unigram vector"
positive=ngramGenerator.mostFreqList('../data/positive_processed.csv',UNIGRAM_SIZE) # add as needed 
negative=ngramGenerator.mostFreqList('../data/negative_processed.csv',UNIGRAM_SIZE)
neutral=ngramGenerator.mostFreqList('../data/neutral_processed.csv',UNIGRAM_SIZE)


total=positive+negative+neutral # total unigram vector
#print len(total)
#total=[]
 
def mapTweet(tweet,sentiWordnet,emoDict,unigram,slangs):
    out=[]
    line=preprocessing.processTweet(tweet,stopWords,slangs)
   
#    p=polarity.polarity(line,sentiWordnet)
    p=polarity.posPolarity(line,sentiWordnet)
   
    out.extend([float(p[0]),float(p[1]),float(p[2])]) # aggregate polarity for pos neg and neutral here neutral is stripped
#    pos=polarity.posFreq(line,sentiWordnet)
    out.extend(p[7:]) # frequencies of pos 

#    out.extend([float(pos['v']),float(pos['n']),float(pos['a']),float(pos['r'])]) # pos counts inside the tweet
    out.append(float(features.emoticonScore(line,emoDict))) # emo aggregate score be careful to modify weights
    out.append(float(len(features.hashtagWords(line))/40)) # number of hashtagged words
    out.append(float(len(line)/140)) # for the length
    out.append(float(features.upperCase(line))) # uppercase existence : 0 or 1
    out.append(float(features.exclamationTest(line)))
    out.append(float(line.count("!")/140))
    out.append(float((features.questionTest(line))))
    out.append(float(line.count('?')/140))
    out.append(float(features.freqCapital(line)))
    for w in unigram:  # unigram
            if (w in line):
                out.append(float(1))
            else:
                out.append(float(0))
    return out

# 
def modiKNN(k,z): # aggregate negative and positive scores for knn
    l=z[0:k]
    l.append(z[k]+z[k+1])
    l.extend(z[k+2:])
    return l

def modiSVM(k,z): # remove neutral polarity score 
    l=z[0:k]
    l.extend(z[k+1:])
    if (len(l)>UNIGRAM_SIZE):
        l=l[:-UNIGRAM_SIZE] # remove neutral unigram words 
    return l

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
        print str(kpos/61)+"% Loading..."
    f.close()
    
    f=open(neufilename,'r')
    line=f.readline()
    while line:
        kneu=kneu+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neulabel))
        line=f.readline()
        print str((kneu+2000)/61)+"% Loading..."
    f.close()
    
    f=open(negfilename,'r')
    line=f.readline()
    while line:
        kneg=kneg+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neglabel))
        line=f.readline()
        print str((kneg+4000)/61)+"% Loading..."
    f.close()
    return vectors,labels


def trainKNN(X,Y,n): # number of neighbors
    clf = neighbors.KNeighborsClassifier(n)
    clf.fit(X,Y)
    return clf

def trainSVM(X,Y,knel):
    clf=svm.SVC(kernel=knel) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf

def predictOneKNN(tweet,model,scaler,normalizer): # test a tweet against a built model KNN or SVM not both 
    z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs) # mapping
    z=modiKNN(0,z)
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return float(model.predict([z]).tolist()[0]) # transform nympy array to list 

def predictOneSVM(tweet,model,scaler,normalizer): # test a tweet against a built model KNN or SVM not both sdandardizer normalizer
    z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs) # mapping
    z=modiSVM(2,z)
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return float(model.predict([z]).tolist()[0]) # transform nympy array to list 

def predictTwo(tweet,knn_model,svm_model): # perform two step classification : KNN then SVM
    z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs) # mapping
    obj=float(predictOneKNN(tweet,knn_model,s1,n1))
    if obj==0.0: # neutral
        return 2.0
    else: # tweet classified as subjective ==> SVM for pos vs neg classification
        subj=float(predictOneSVM(tweet,svm_model,s2,n2))
        return subj

def predictVector(z,knn_model,svm_model,s1,n1,s2,n2): # from a vector predict label using hybrid, to use for cross validation
    result=[]
    for x in z:
        x1=modiKNN(0,x)
        x1_scaled=s1.transform(x1)
        x1=n1.transform([x1_scaled])
        x1=x1[0].tolist()
        p1=float(knn_model.predict([x1]).tolist()[0])
        if p1==0:
            result.append(2.0)
        else:
            x2=modiSVM(2,x)
            x2_scaled=s2.transform(x2)
            x2=n2.transform([x2_scaled])
            x2=x2[0].tolist()
            result.append(float(svm_model.predict([x2]).tolist()[0]))
    return result

#def predictFile
def predictFile(filename,knn_model,svm_model): # function to load test file in the csv format : sentiment,tweet 
    f=open(filename,'r')
    fo=open(filename+".result",'w')
    line=f.readline()
    while line:
        tweet=line[:-1]
        nl=predictTwo(tweet,knn_model,svm_model)
    
        fo.write(r'"'+str(nl)+r'","'+tweet+r'"\n')
        line=f.readline()
    
    f.close()
    fo.close()
    print "Tweets are classified . The result is in "+filename+".result"

# Test dataset classification
def testFile(filename,knn_model,svm_model): # function to load test file in the csv format : sentiment,tweet 
    f=open(filename,'r')
    fo=open(filename+".result",'w')
    line=f.readline()
    labels=[]
    newLabels=[]
    p=0 # precision
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]
        nl=predictTwo(tweet,knn_model,svm_model)
        newLabels.append(nl)
        if nl != s:
            p=p+1
    
        labels.append(s)
        fo.write("new label :"+str(nl)+" old label :"+str(s)+" tweet : "+tweet+'\n')
        line=f.readline()
    if (len(labels) != 0):
        p=p/len(labels)
    f.close()
    fo.close()
    print "Tweets in test file are classified . The result is in "+filename+".result"
    print "Accuracy over test file is : "+str(p)   # for now 

# 5 fold cross validation test
def validateHybrid(X,Y,n,knel,c):
    scores=[] # list accuracy values for each fold
    #folds=[k*int(len(X)/5) for k in range(1,5)]
    for k in range(1,6):
        err=0 #aggregate error 
        X_test=X[(k-1)*int(len(X)/5):k*int(len(X)/5)]
        Y_test=Y[(k-1)*int(len(X)/5):k*int(len(X)/5)]
        X_train=X[:(k-1)*int(len(X)/5)]+X[(k)*int(len(X)/5):]
        Y_train=Y[:(k-1)*int(len(X)/5)]+Y[(k)*int(len(X)/5):]
        KNN_MODEL,SVM_MODEL,s1,n1,s2,n2=buildHybrid(X,Y,n,knel,c)
        z=predictVector(X_test,KNN_MODEL,SVM_MODEL,s1,n1,s2,n2)
        for j in range(0,len(z)):
            if (z[j] != Y_test[j]):
                err=err+1
        scores.append(1-err/len(Y_test))
    scores=np.array(scores)

    print("Accuracy of the hybrid model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 







# Buildng hybrid model function from loaded data
def buildHybrid(X,Y,n,knel,c): # n for neighbors, knel for kernel function 
    X_KNN=[]
    Y_KNN=[]
    X_SVM=[]
    Y_SVM=[]

    for w in Y:
        if w == 2.0:
            Y_KNN.append(0.0)
        else:
            Y_KNN.append(1.0) # for subjective

    for z in X:
        X_KNN.append(modiKNN(0,z))
    
    # features standardization 
    X_KNN_scaled=pr.scale(np.array(X_KNN))
    scalerKNN = pr.StandardScaler().fit(X_KNN) # to use later for testing data scaler.transform(X) 

    # features Normalization
    X_KNN_normalized = pr.normalize(X_KNN_scaled, norm='l2') # l2 norm
    normalizerKNN = pr.Normalizer().fit(X_KNN_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

    X_KNN=X_KNN_normalized.tolist()

    x=np.array(X_KNN)
    y=np.array(Y_KNN)
    clf = neighbors.KNeighborsClassifier(n)
    scores = cross_validation.cross_val_score(clf, x, y, cv=5)
    #print scores # the precision for five iterations
    #print("Accuracy of the knn model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 
    
    KNN_MODEL=trainKNN(X_KNN,Y_KNN,n) # 5 neighbors

    for i in range(0,len(Y)):
        if (Y[i] != 2.0): # no neutral in the svm model
            X_SVM.append(modiSVM(2,X[i]))
            Y_SVM.append(Y[i])


    X_SVM_scaled=pr.scale(np.array(X_SVM))
    scalerSVM = pr.StandardScaler().fit(X_SVM) # to use later for testing data scaler.transform(X) 

    # features Normalization
    X_SVM_normalized = pr.normalize(X_SVM_scaled, norm='l2') # l2 norm
    normalizerSVM = pr.Normalizer().fit(X_SVM_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

    X_SVM=X_SVM_normalized.tolist()

    x=np.array(X_SVM)
    y=np.array(Y_SVM)
    clf = svm.SVC(kernel=knel, C=c)
    scores = cross_validation.cross_val_score(clf, x, y, cv=5)
    #print scores # the precision for five iterations
    #print("Accuracy of the svm model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 

    SVM_MODEL=trainSVM(X_SVM,Y_SVM,knel) # change to poly if needed 
    return KNN_MODEL, SVM_MODEL,scalerKNN,normalizerKNN,scalerSVM,normalizerSVM # return both models



# loading training data
print "Loading training data"
#X,Y=loadMatrix('../data/positive_processed.csv','../data/neutral_processed.csv','../data/negative_processed.csv','4','2','0')
X,Y=loadMatrix('../data/small_positive_processed.csv','../data/small_neutral_processed.csv','../data/small_negative_processed.csv','4','2','0')

# features standardization 
X_scaled=pr.scale(np.array(X))
scaler = pr.StandardScaler().fit(X) # to use later for testing data scaler.transform(X) 

# features Normalization
X_normalized = pr.normalize(X_scaled, norm='l2') # l2 norm
normalizer = pr.Normalizer().fit(X_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

X=X_normalized
X=X.tolist()



# validation step 
print "Performing 5 fold cross validation on the model "
validateHybrid(X,Y,N_NEIGHBORS,KERNEL_FUNCTION,C_PARAMETER)
# Building Model
print "Initializing model ..."
KNN_MODEL,SVM_MODEL,s1,n1,s2,n2=buildHybrid(X,Y,N_NEIGHBORS,KERNEL_FUNCTION,C_PARAMETER)



# test dataset classification, uncomment the next line to perform the test 
print "Testing model with test dataset ..."
#testFile('../data/test_dataset.csv',KNN_MODEL,SVM_MODEL)
testFile('../data/small_test_dataset.csv',KNN_MODEL,SVM_MODEL)

print "Model Built . Want to classify a tweet ? ..."


user_input=raw_input("Write a tweet to test or a file path for bulk classification with Hybrid model. press q to quit\n")
while user_input!='q':
    try:
        predictFile(user_input,KNN_MODEL,SVM_MODEL)
    except:
        print "sentiment : "+str(predictTwo(user_input,KNN_MODEL,SVM_MODEL))
        user_input=raw_input("Write a tweet to test or a file path for bulk classification . press q to quit\n")

# the end !
