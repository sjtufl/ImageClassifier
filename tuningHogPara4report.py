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

def drawPillar(preErrArray):     
    n_groups = 1;       
    perdic42 = preErrArray[0]
    perdic44 = preErrArray[1]
    perdic82 = preErrArray[2]    
         
    fig, ax = plt.subplots()    
    index = np.arange(n_groups)    
    bar_width = 0.35    
         
    opacity = 0.4    
    rects1 = plt.bar(index, perdic42, bar_width,alpha=opacity, color='b',label=    'cell 4*4 block 2*2')    
    rects2 = plt.bar(index + bar_width, perdic44, bar_width,alpha=opacity,color='r',label='cell 4*4 block 4*4')
    rects3 = plt.bar(index + bar_width*2, perdic82, bar_width, alpha=opacity, color='g', label="cell 8*8 block 2*2")    
         
    plt.xlabel('Cell and block size of HOG descriptor')    
    plt.ylabel('Prediction error')     
    plt.xticks(index + bar_width, ('x'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("lda-hog-cellblock.png")

if __name__ == "__main__":
    t0 = time.time()

    #para = "orientation"
    para = "cellblock"

    if para is "orientation":
        #orientations
        cnt = 0
        train_feat_path_o = ["./data/features/train-o3", "./data/features/train-o6", "./data/features/train", "./data/features/train-o12", "./data/features/train-o15"]
        test_feat_path_o = ["./data/features/test-o3", "./data/features/test-o6", "./data/features/test", "./data/features/test-o12", "./data/features/test-o15"]
        preErrArray = []
        for train_feat_path in train_feat_path_o:
            fds = []
            labels = []

            for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
                data = joblib.load(feat_path)
                fds.append(data[:-1])
                labels.append(data[-1])

            total = 0
            num = 0
            clf = LinearDiscriminantAnalysis()
            print "Training a LDA Classifier Model, Orientations=%d" % ((cnt+1)*3)
            clf.fit(fds, labels)
            for feat_path in glob.glob(os.path.join(test_feat_path_o[cnt], '*.feat')):
                total += 1
                data_test = joblib.load(feat_path)
                data_test_feat = data_test[:-1].reshape((1, -1))
                result = clf.predict(data_test_feat)
                if int(result) == int(data_test[-1]):
                    num += 1
            acc = float(num)/total
            preErrArray.append(1-acc)
            print 'The LDA classification accuracy is %f' % acc
            cnt += 1

        t1 = time.time()
        print 'The cost of time is %f' % (t1-t0)

        oArray = [3,6,9,12,15]
        #draw the picture c-predict err
        plt.xlabel("Orientations of HOG descriptor")
        plt.ylabel("Prediction error")
        plt.plot(oArray, preErrArray, 'o-', label="HOG")
        plt.savefig("lda-hog-orient.png")

    if para is "cellblock":
        #cells and blocks
        cnt = 0
        train_feat_path_cb = ["./data/features/train42", "./data/features/train44", "./data/features/train"]
        test_feat_path_cb = ["./data/features/test42", "./data/features/test44", "./data/features/test"]
        preErrArray = []
        for train_feat_path in train_feat_path_cb:
            fds = []
            labels = []

            for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
                data = joblib.load(feat_path)
                fds.append(data[:-1])
                labels.append(data[-1])

            total = 0
            num = 0
            clf = LinearDiscriminantAnalysis()
            print "Training a LDA Classifier Model, cellblock path = %s" % train_feat_path_cb[cnt] 
            clf.fit(fds, labels)
            for feat_path in glob.glob(os.path.join(test_feat_path_cb[cnt], '*.feat')):
                total += 1
                data_test = joblib.load(feat_path)
                data_test_feat = data_test[:-1].reshape((1, -1))
                result = clf.predict(data_test_feat)
                if int(result) == int(data_test[-1]):
                    num += 1
            acc = float(num)/total
            preErrArray.append(1-acc)
            print 'The LDA classification accuracy is %f' % acc
            cnt += 1

        t1 = time.time()
        print 'The cost of time is %f' % (t1-t0)

        #draw the picture c-predict err
        drawPillar(preErrArray)