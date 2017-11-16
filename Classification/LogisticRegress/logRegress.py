#encoding = utf-8

'''
    Logistic 回归
    梯度上升 （每次 更新 都是 加）
'''

import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    f = open("testSet.txt",'r')
    for line in f.readlines():
        lineArr = line.strip().split()                                  # strip 去掉 前后空格和换行，  然后 切分
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])      # 1.0 常数项的值
        labelMat.append(int(lineArr[2]))                                # 没有加[]  表示一个行向量
    return dataMat,labelMat

def sigmoid(inX):
    return np.longfloat( 1.0/(1+ np.exp(- inX)) )                             # 这里是 np.exp 不是 math.exp（不能对矩阵进行操作）

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()                # 行向量 转化为 列向量
    m,n = np.shape(dataMatrix)
    alpha = 0.001                                   # 向目标移动的步长
    maxCycles = 500                                 # 循环次数
    weight = np.ones((n,1))                          # 初始化 一个 n * 1 的 列向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weight)                # h : m ,n * n,1 = m*1
        error = (labelMat - h)
        weight = weight + alpha * dataMatrix.transpose() * error
    return weight



# dataArr,labelMat = loadDataSet()
# print(gradAscent(dataArr,labelMat))


'''
    画出决策边界
'''
def plotBestFit():
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    weights = weights.getA1()                                   # getA1() 返回一个扁平（一维）的数组（ndarray）
    print(weights)
    n = len(dataMat)
    xcord1 = [] ; ycord1 = []                                   # label = 1  的 坐标点
    xcord2 = [] ; ycord2 = []
    for i in range(n):
        if (int(labelMat[i]) == 1):
            xcord1.append(dataMat[i][1]) ; ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1]) ; ycord2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30 , c = "red", marker='s')
    ax.scatter(xcord2,ycord2,s = 30, c = 'green')
    x = np.arange(-5.0, 5.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel("X2")
    plt.show()


'''
    改进的随机梯度上升算法
'''
def stockGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones((n,1))
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) +0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    prob = prob[0]
    if prob > 0.5 :
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):                              #  22列   21 个特征
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))      #最后一个是标签
    trainWeights = stockGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum  = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the averafe error rate is : %f" % (numTests, errorSum/float(numTests)))

multiTest()
