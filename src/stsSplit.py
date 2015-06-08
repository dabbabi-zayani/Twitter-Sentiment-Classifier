import preprocessing
stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')

f=open('../data/sts/positive_sample.csv','r')
fo=open('../data/positive_processed.csv','w')

line=f.readline()
while line:
    a=line.split(r'","')
    b=a[5][:-1]
    c=preprocessing.processTweet(b,stopWords,slangs)
    
    d=preprocessing.removeStopWords(c,stopWords)
    
    fo.write(d+'\n')
    line = f.readline()

f.close()
fo.close()

print "positive samples processed"

f=open('../data/sts/negative_sample.csv','r')
fo=open('../data/negative_processed.csv','w')
line=f.readline()
while line:
    a=line.split(r'","')
    b=a[5][:-1]
    c=preprocessing.processTweet(b,stopWords,slangs)
    d=preprocessing.removeStopWords(c,stopWords)
    fo.write(d+'\n')
    line = f.readline()

f.close()
fo.close()

print "negative sample processed"

f=open('../data/sts/neutral_sample.csv','r')
fo=open('../data/neutral_processed.csv','w')
line=f.readline()
while line:
    a=line.split(r'","')
    b=a[4][:-1]
    c=preprocessing.processTweet(b,stopWords,slangs)
    d=preprocessing.removeStopWords(c,stopWords)
    fo.write(d+'\n')
    line = f.readline()

f.close()
fo.close()

print "neutral sample processed"
