# predictor
import sys
import os

classifiers=['svm','knn','hybrid']
try:
    choice=str(sys.argv[1])
    if (choice not in classifiers or len(sys.argv) > 2):
        print "Usage : python predictor.py <classifier> : possible classifiers are : svm, knn or hybrid"
        sys.exit()

    command="python "+str(choice)+".py"
    os.system(command)

except:
    sys.exit()

