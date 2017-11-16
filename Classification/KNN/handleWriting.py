#encoding = utf-8

'''
    手写数字  识别 ，将数字图 转为 32 * 32 的黑白图，如果需要用我们之前的 kNN里面的一维 向量 就需要将图形转为 1 * 1024 的形式再做比较
'''

import numpy as np
import os
from Classification.KNN.kNN import classify

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    f = open(filename)
    for i in  range(32):
        line = f.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(line[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    rows = len(trainingFileList)
    trainingMat = np.zeros((rows,1024))
    for i in range(rows):
        fileName = trainingFileList[i].split(".")[0]
        classNumber = int(fileName.split("_")[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector("trainingDigits/"+trainingFileList[i])
    testFileList = os.listdir("testDigits")
    test_rows = len(testFileList)
    errorCount = 0.0
    for i in range(test_rows):
        fileName = testFileList[i].split(".")[0]
        classNumber = int(fileName.split("_")[0])
        test_vector = img2vector("testDigits/"+testFileList[i])
        classifierResult = classify(test_vector,trainingMat,hwLabels,3)
        if(classNumber != classifierResult):
            errorCount += 1.0
    print("\n The total number of errors is %d" % errorCount)
    print("\n The total error rate is %f" % (errorCount/float(test_rows)))

if __name__ =="__main__":
    handwritingClassTest()
