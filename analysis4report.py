#!/usr/bin/env python
#encoding:utf-8
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time
from config import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t0 = time.time()
    #clf_type = 'LIN_SVM'
    clf_type = 'KNN'
    #clf_type = 'LDA'
    fds = []
    labels = []
    #num = 0
    #total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])

    if clf_type is 'LIN_SVM':
        cArray = [float(0.2), 0.4, 0.6, 0.8, 1.0, 2, 3, 4, 5, 7, 9, 10, 12.6]
        preErrArray = []
        accRateArray = []
        for c in cArray:
            num = 0
            total = 0
            clf = LinearSVC(C=c)
            print "Training a Linear SVM Classifier, C=%f" % c
            clf.fit(fds, labels)
            for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
                total += 1
                data_test = joblib.load(feat_path)
                data_test_feat = data_test[:-1].reshape((1, -1))
                result = clf.predict(data_test_feat)
                if int(result) == int(data_test[-1]):
                    num += 1
            acc = float(num)/total
            accRateArray.append(acc)
            preErrArray.append(1-acc)
            print 'The SVM classification accuracy is %f' % acc
        print "Job finished."
        t1 = time.time()
        print 'The cost of time is %f' % (t1-t0)
        #draw the picture c-predict err
        plt.xlabel("C-Value")
        plt.ylabel("Prediction Error")
        plt.plot(cArray, preErrArray, 'o-', label="HOG-[2*2][8*8]9")
        #plt.show()
        plt.savefig("svm.png")

    if clf_type is 'KNN':
        kArray = [2,4,6,8,10,12,15,18,24,50,100,150]
        preErrArray = []
        accRateArray = []
        for k in kArray:
            num = 0
            total = 0
            clf = KNeighborsClassifier(n_neighbors = k)
            clf.fit(fds, labels)
            for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
                total += 1
                data_test = joblib.load(feat_path)
                data_test_feat = data_test[:-1].reshape((1, -1))
                result = clf.predict(data_test_feat)
                if int(result) == int(data_test[-1]):
                    num += 1
            acc = float(num)/total
            accRateArray.append(acc)
            preErrArray.append(1-acc)
            print 'The KNN classification accuracy is %f' % acc
        print "Job finished."
        t1 = time.time()
        print 'The cost of time is %f' % (t1-t0)
        #draw the picture k-predict err
        plt.xlabel("The number of neighbors K")
        plt.ylabel("Prediction Error")
        plt.plot(kArray, preErrArray)
        #plt.show()
        plt.savefig("knn.png")

    if clf_type is 'LDA':
        total = 0
        num = 0
        clf = LinearDiscriminantAnalysis()
        print "Training a LDA Classifier Model..."
        clf.fit(fds, labels)
        if not os.path.isdir(os.path.split(lda_model_path)[0]):
            #print "Path doesn't exist!"
            os.makedirs(os.path.split(lda_model_path)[0])
        joblib.dump(clf, lda_model_path)
        print "LDA Classifier saved to {}".format(lda_model_path)
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
                #print "result: %d, Num: %d, total: %d" % (int(result), num, total)
        rate = float(num)/total
        t1 = time.time()
        print 'The LDA classification accuracy is %f' % rate
        print 'The cost of time is %f' % (t1-t0)