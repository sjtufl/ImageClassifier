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

if __name__ == "__main__":
    t0 = time.time()
    #clf_type = 'LIN_SVM'
    #clf_type = 'KNN'
    #clf_type = 'LDA'
    clf_type = 'LDA_TEST'
    fds = []
    labels = []
    num = 0
    total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])

    if clf_type is 'LIN_SVM':
        clf = LinearSVC()
        print "Training a Linear SVM Classifier."
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(svm_model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, svm_model_path)
        # clf = joblib.load(model_path)
        print "SVM Classifier saved to {}".format(svm_model_path)
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                #print "result: %d, Num: %d" % (int(result), num)
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print 'The classification accuracy is %f'%rate
        print 'The cost of time is :%f'%(t1-t0)

    if clf_type is 'KNN':
        clf = KNeighborsClassifier()
        print "Training a KNN Classifier Model..."
        clf.fit(fds, labels)
        if not os.path.isdir(os.path.split(knn_model_path)[0]):
            #print "Path doesn't exist!"
            os.makedirs(os.path.split(knn_model_path)[0])
        joblib.dump(clf, knn_model_path)
        print "KNN Classifier saved to {}".format(knn_model_path)
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
                print "result: %d, Num: %d, total: %d" % (int(result), num, total)
        rate = float(num)/total
        t1 = time.time()
        print 'The KNN classification accuracy is %f' % rate
        print 'The cost of time is %f' % (t1-t0)

    if clf_type is 'LDA':
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

    if clf_type is 'LDA_TEST':
        clf = LinearDiscriminantAnalysis()
        print "Test a LDA Classifier Model..."
        clf = joblib.load(lda_model_path)
        print "Model Loaded!"
        for feat_path in glob.glob(os.path.join("./data/features/test_pic", '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test.reshape((1, -1))
            result = clf.predict(data_test_feat)
            print "result: %d, total: %d" % (int(result), total)

    if clf_type is 'SVM_TEST':
        clf = LinearSVC()
        print "Test a SVM classifier Model..."
        clf = joblib.load(svm_model_path)
        print "Model loaded!"
        for feat_path in glob.glob(os.path.join("./data/features/test_pic", "*.feat")):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test.reshape((1, -1))
            result = clf.predict(data_test_feat)
            print "result: %d, totgal: %d" % (int(result), total)