# A twitter sentiment classifier based on Support Vector Machines and K nearest neighbors algorithms
Overall decription
-------
As undestood from the title, this repository contains sources codes (src folder) , datasets (data folder) and useful resources for twitter sentiment analysis (resources folder).<br />
The training dataset is split into 3 files containing a processed version of tweets in the three classes : positive (data/used/positive1.csv), negative (data/used/negative1.csv) and neutral (data/used/neutral1.csv) <br />

The training dataset is collected SemEval challenge ( http://alt.qcri.org/semeval2014/task9/index.php?id=data-and-tools ), STS gold(http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)  and Sanders dataset (http://www.sananalytics.com/lab/twitter-sentiment) . The testing dataset is from STS-Gold (http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) <br />

The test datasets are STS_Test (data/test_dataset.csv) and 100 3cixty reviews in (data/3cixty/3cixty_test_dataset.csv) <br />  
In the src folder :<br />
1) svm.py : svm classifier <br />
2) knn.py : knn classifier <br />
3) hybrid.py : 2 step classification : knn for objectivity/subjectivity test, svm for polarity test <br />
Emoticons dictionnary, Stop Words list, SentiWordnet 3.0.1, AFINN , and a slang dictionnary are in the resources folder.<br />
<br />
Requirements
-------

The classifier works for python 2.6 and 2.7 <br />
To use these algorithms you should install : sklearn 0.14 version (http://scikit-learn.org/dev/index.html) , numpy (http://www.numpy.org/), nltk 3 with full packages using nltk.download() instruction in python <br />
<br />

Running the classifiers
-------
Runnig any classifier of the mentioned above is done as by executing the classifier.py script as follow  : <br />
Usage : python predictor.py classifier_choice <br />
Available classifiers are : svm, knn or hybrid<br />

N.B : The class labels are real values and are as follow : positive : 4.0, negative : 0.0 and neutral 2.0 <br />


Thank you .  <br />

