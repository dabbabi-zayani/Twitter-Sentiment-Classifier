from __future__ import division
from sklearn import neighbors
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing as pr
from sklearn import metrics



import numpy as np

import features
import polarity
import ngramGenerator
import preprocessing


# User input for model parameters
N_NEIGHBORS=10  # number of neighbors for KNN
KERNEL_FUNCTION='linear'  # kernel function for SVM
C_PARAMETER=0.2
UNIGRAM_SIZE=3000



print "Initializing dictionnaries"
stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')
afinn=polarity.loadAfinn('../resources/afinn.txt')
emoticonDict=features.createEmoticonDictionary("../resources/emoticon.txt")

print "Bulding unigram vector"
positive=ngramGenerator.mostFreqList('../data/used/positive1.csv',UNIGRAM_SIZE) # add as needed 
negative=ngramGenerator.mostFreqList('../data/used/negative1.csv',UNIGRAM_SIZE)
neutral=ngramGenerator.mostFreqList('../data/used/neutral1.csv',UNIGRAM_SIZE)


for w in positive:
    if w in negative+neutral : 
        positive.remove(w)

for w in negative:
    if w in positive+neutral : 
        negative.remove(w)

for w in neutral:
    if w in negative+positive : 
        neutral.remove(w)

# equalize unigrams sizes 
m=min([len(positive),len(negative),len(neutral)])       

positive=positive[0:m-1]
negative=negative[0:m-1]
neutral=neutral[0:m-1]

 
def mapTweet(tweet,afinn,emoDict,positive,negative,neutral,slangs):
    out=[]
    line=preprocessing.processTweet(tweet,stopWords,slangs)
    p=polarity.afinnPolarity(line,afinn)
    out.append(p)
    out.append(float(features.emoticonScore(line,emoDict))) # emo aggregate score be careful to modify weights
    out.append(float(len(features.hashtagWords(line))/40)) # number of hashtagged words
    out.append(float(len(line)/140)) # for the length
    out.append(float(features.upperCase(line))) # uppercase existence : 0 or 1
    out.append(float(features.exclamationTest(line)))
    out.append(float(line.count("!")/140))
    out.append(float((features.questionTest(line))))
    out.append(float(line.count('?')/140))
    out.append(float(features.freqCapital(line)))
    u=features.scoreUnigram(line,positive,negative,neutral)
    out.extend(u)
    return out

# 
def modiKNN(z): # aggregate negative and positive scores for knn
    l=z[0:len(z)-3]
    l.append((z[len(z)-3]+z[len(z)-2])/2) # pos and neg 
    l.append(z[len(z)-1]) # neutral score 
    return l

def modiSVM(z): # remove neutral polarity score 
    l=z[0:len(z)-1]
    return l

# load matrix
def loadMatrix(posfilename,neufilename,negfilename,poslabel,neulabel,neglabel):
    vectors=[]
    labels=[]
    print "Loading training dataset..."
    f=open(posfilename,'r')
    kpos=0
    kneg=0
    kneu=0
    line=f.readline()
    while line:
        
        try:
            kpos+=1
            z=mapTweet(line,afinn,emoticonDict,positive,negative,neutral,slangs)
            vectors.append(z)
            labels.append(float(poslabel))
        except:
            None
        line=f.readline()
#        print str(kpos)+"positive lines loaded : "+str(z)
    f.close()
    
    f=open(neufilename,'r')
    line=f.readline()
    while line:
        try:
            kneu=kneu+1
            z=mapTweet(line,afinn,emoticonDict,positive,negative,neutral,slangs)
            vectors.append(z)
            labels.append(float(neulabel))
        except:
            None
        line=f.readline()
#        print str(kneu)+"neutral lines loaded : "+str(z)
    f.close()
    
    f=open(negfilename,'r')
    line=f.readline()
    while line:
        try:
            kneg=kneg+1
            z=mapTweet(line,afinn,emoticonDict,positive,negative,neutral,slangs)
            vectors.append(z)
            labels.append(float(neglabel))
        except:
            None
        line=f.readline()
#        print str(kneg)+"negative lines loaded : "+str(z)
    f.close()
    print "Loading done."
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
    z=mapTweet(tweet,afinn,emoticonDict,positive,negative,neutral,slangs) # mapping
    z=modiKNN(z)
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return float(model.predict([z]).tolist()[0]) # transform nympy array to list 

def predictOneSVM(tweet,model,scaler,normalizer): # test a tweet against a built model KNN or SVM not both sdandardizer normalizer
    z=mapTweet(tweet,afinn,emoticonDict,positive,negative,neutral,slangs) # mapping
    z=modiSVM(z)
    z_scaled=scaler.transform(z)
    z=normalizer.transform([z_scaled])
    z=z[0].tolist()
    return float(model.predict([z]).tolist()[0]) # transform nympy array to list 

def predictTwo(tweet,knn_model,svm_model): # perform two step classification : KNN then SVM
    z=mapTweet(tweet,afinn,emoticonDict,positive,negative,neutral,slangs) # mapping
    obj=float(predictOneKNN(tweet,knn_model,s1,n1))
    if obj==0.0: # neutral
        return 2.0
    else: # tweet classified as subjective ==> SVM for pos vs neg classification
        subj=float(predictOneSVM(tweet,svm_model,s2,n2))
        return subj

def predictVector(z,knn_model,svm_model,s1,n1,s2,n2): # from a vector predict label using hybrid, to use for cross validation
    result=[]
    for x in z:
        x1=modiKNN(x)
        x1_scaled=s1.transform(x1)
        x1=n1.transform([x1_scaled])
        x1=x1[0].tolist()
        p1=float(knn_model.predict([x1]).tolist()[0])
        if p1==0:
            result.append(2.0)
        else:
            x2=modiSVM(x)
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
    fo=open(filename+".hybrid_result",'w')
    line=f.readline()
    labels=[]
    newLabels=[]
    mispos=0
    misneg=0
    misneu=0

    p=0 # precision
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]
        nl=predictTwo(tweet,knn_model,svm_model)
        newLabels.append(nl)
        fo.write(r'"'+str(s)+r'","'+tweet+r'","'+str(nl)+r'"'+"\n")

        if nl != s:
            p=p+1
    
        labels.append(s)
        line=f.readline()
    if (len(labels) != 0):
        p=p/len(labels)
        p=1-p
    f.close()
    fo.close()
    print "Tweets in test file are classified . The result is in "+filename+".hybrid_result"
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
        KNN_MODEL,SVM_MODEL,s1,n1,s2,n2=buildHybrid(X_train,Y_train,n,knel,c)
        z=predictVector(X_test,KNN_MODEL,SVM_MODEL,s1,n1,s2,n2)
        
        scores.append(metrics.accuracy_score(Y_test,np.array(z)))
    scores=np.array(scores)

    print("Accuracy of the hybrid model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 
    return scores.mean()





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
        X_KNN.append(modiKNN(z))
    
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
            X_SVM.append(modiSVM(X[i]))
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
X,Y=loadMatrix('../data/used/positive1.csv','../data/used/neutral1.csv','../data/used/negative1.csv','4','2','0')
#X,Y=loadMatrix('../data/small_positive_processed.csv','../data/small_neutral_processed.csv','../data/small_negative_processed.csv','4','2','0')

# features standardization 
X_scaled=pr.scale(np.array(X))
scaler = pr.StandardScaler().fit(X) # to use later for testing data scaler.transform(X) 

# features Normalization
X_normalized = pr.normalize(X_scaled, norm='l2') # l2 norm
normalizer = pr.Normalizer().fit(X_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test

X=X_normalized
X=X.tolist()



# validation step 
print "Optimizing "
C=[0.01*i for i in range(1,2)]
N=[i for i in range(10,11)]
ACC=0.0
best_acc=0.0
iter=0
for c in C:
    for n in N:
        print "C parameter : %f, Neighbors %d" %(c,n)
        ACC=validateHybrid(X,Y,n,KERNEL_FUNCTION,c)
        if (ACC > best_acc):
            N_NEIGHBORS=n
            C_PARAMETER=c
            best_acc=ACC

print "Model optimized "
print "best c : %f, best n : %d , best accuracy : %f" %(C_PARAMETER,N_NEIGHBORS,best_acc)
# Building Model
print "Initializing model ..."
KNN_MODEL,SVM_MODEL,s1,n1,s2,n2=buildHybrid(X,Y,N_NEIGHBORS,KERNEL_FUNCTION,C_PARAMETER)



# test dataset classification, uncomment the next line to perform the test 
print "Testing model with test dataset ..."
testFile('../data/test_dataset.csv',KNN_MODEL,SVM_MODEL)
#testFile('../data/small_test_dataset.csv',KNN_MODEL,SVM_MODEL)

print "Model Built . Want to classify a tweet ? ..."


user_input=raw_input("Write a tweet to test or a file path for bulk classification with Hybrid model. press q to quit\n")
while user_input!='q':
    try:
        predictFile(user_input,KNN_MODEL,SVM_MODEL)
        print "labels are : 4.0 for positive, 2.0 for neutral and 0.0 for negative tweets"
        user_input=raw_input("Write a tweet to test or a file path for bulk classification . press q to quit\n")
    except:
        print "sentiment : "+str(predictTwo(user_input,KNN_MODEL,SVM_MODEL))
        print "labels are : 4.0 for positive, 2.0 for neutral and 0.0 for negative tweets"
        user_input=raw_input("Write a tweet to test or a file path for bulk classification . press q to quit\n")

# the end !

