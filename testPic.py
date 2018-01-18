from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from sklearn.externals import joblib
# To read image file and save image feature descriptions
import os
import time
import glob
import cPickle
from config import *


def getData(filePath):
	allData = []
	allFileName = []
	for file in os.listdir(filePath):
		fileName = file.split('.')[0]
		img = Image.open(filePath + '/' + file)
		dst = img.resize((32, 32))
		new = dst.convert('RGB')
		# r, g, b = new.split()
		data = new.getdata()
		data = np.matrix(data, dtype='int')
		newData = np.reshape(data, (3, 32 * 32))
		allData.append(newData)
		allFileName.append(fileName)
	testPic = np.reshape(allData, (2175, 3, 32 * 32))
	fileNames = np.reshape(allFileName, (2175, 1))	
	TestPicData = zip(testPic, fileNames)
	return TestPicData

def getFeat(TestPicData):
    for data in TestPicData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        fd = hog(gray, orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2], visualise=False, transform_sqrt=True)
        #fd = hog(gray, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        # fd = np.concatenate((fd, data[1]))
        # filename = list(data[2])
        filename = list(data[1])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/test_pic/', fd_name)
        joblib.dump(fd, fd_path)
    print "TestPic features are extracted and saved."

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

if __name__ == '__main__':
    t0 = time.time()
    filePath = './data/testData'
    TestPicData = getData(filePath)
    getFeat(TestPicData)
    t1 = time.time()
    print "Features are extracted and saved."
    print 'The cast of time is:%f'%(t1-t0)


# plt.figure("test image-0000.jpg")
# plt.subplot(2,2,1), plt.title('32x32')
# plt.imshow(new), plt.axis('off')
# plt.subplot(2,2,2), plt.title('32x32 r')
# plt.imshow(r), plt.axis('off')
# plt.subplot(2,2,3), plt.title('32x32 g')
# plt.imshow(g), plt.axis('off')
# plt.subplot(2,2,4), plt.title('32x32 b')
# plt.imshow(b), plt.axis('off')
# plt.show()