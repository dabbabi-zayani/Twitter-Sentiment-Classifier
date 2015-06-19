# A twitter sentiment classifier based on Support Vector Machines and K nearest neighbors
In the src folder :<br />
1) svm.py : svm classifier <br />
2) knn.py : knn classifier <br />
3) hybrid.py : 2 step classification : knn for objectivity/subjectivity test, svm for polarity test <br />
<br />
The classifier works for python 2.6 and 2.7 <br />
To use these algorithms you should install : sklearn 0.14 version (http://scikit-learn.org/dev/index.html) , numpy (http://www.numpy.org/), nltk 3 with full packages using nltk.download() instruction in python <br />
<br />
Emoticons dictionnary, Stop Words list, SentiWordnet 3.0.1, afinn ,  and a slang dictionnary are in the resources folder. <br />
Training and testing datasets are in the data folder.  <br />
Tha training dataset is collected SemEval challenge ( http://alt.qcri.org/semeval2014/task9/index.php?id=data-and-tools ), STS gold and Sanders dataset. The testing dataset is from STS-Gold (http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
<br />
Thank you .  <br />
