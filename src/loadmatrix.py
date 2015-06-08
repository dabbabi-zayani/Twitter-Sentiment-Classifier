# script to load training feature vectors from a file
# the first value in the vector is the label , the rest is the features values 

filename='../data/bigVectorTweets.csv' # input file name


def loadMatrix(filename):
    vectors=[]
    labels=[]
    f=open(filename,'r')
    line=f.readline()
    while line:
        
        l=line.split('\t')
        l=l[:-1]
        vectors.append([float(x) for x in l[1:]])
        labels.append(float(l[0]))
        line=f.readline()
    f.close()
    return vectors,labels

v,l=loadMatrix(filename)
#print len(v[0])
#print len(l)
