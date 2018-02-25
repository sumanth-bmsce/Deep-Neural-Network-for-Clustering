from sklearn.cluster import KMeans
import sklearn.datasets
import numpy as np
import csv;
from itertools import groupby
from sklearn import preprocessing

def k_means(data, no_of_clusters):
    data = np.array(data);
    confusion_matrix=[]
    kmeans = KMeans(no_of_clusters, random_state=0).fit_predict(data);
    l1=kmeans[:458];
    l1.sort();
    #print l1;
    l = [len(list(group)) for key, group in groupby(l1)]
    confusion_matrix.append(l);
    #print l;
    max1 = max(l);
    l1=kmeans[458:699];
    l1.sort();
    #print l1;
    l = [len(list(group)) for key, group in groupby(l1)]
    confusion_matrix.append(l);
    #print l;
    max2 = max(l)
    print "Confusion Matrix"
    for i in range(0,2):
        print confusion_matrix[i]
    print("--------------------")
    print ("Accuracy = "+str((float(max1+max2)/699)*100))
    
    
  
    
