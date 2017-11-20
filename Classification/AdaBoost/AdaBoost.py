#encoding = utf-8

'''
    Advantages: 泛化错误率低，无参数调整
    disadvantages:对离群点敏感
    适用数据类型： 数值型和表称型数据
'''

import numpy as np


'''
    简单的数据集
'''

def loadSimpData():
    dataMat = np.mat([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])              # 5*2
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels


''' 根据阈值来比较数据进行分类，大于阈值的放在右边，小于的放在左边'''

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))                             # 5*1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal ] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal ] = -1.0
    return retArray

''' 得到最小误差的 切分特征 切分位置 ，和经过划分后的数据集'''
def bulidStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T                                # 5*1 --> 1*5
    m,n = np.shape(dataMatrix)

    numSteps = 10.0                                                 # 将某一特征值 切分为 10 个部分
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError  = np.inf                                              # 正无穷

    for i in range(n):                                               #对每一个字段进行遍历
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps)+1):                           #对每一个步长进行便利
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)           # [rangeMin-step, rangemin, rangeMin+step ,...,rangeMin+9step,rangeMin+10step=max]
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat ] = 0
                weightedError = D.T * errArr                            # m*1.T  *  m*1 ,对错误分配的部分分配权值

                # print("split : dim {0},thresh {1}, ineq {2}, "
                #       "weightedError : {3}\n".format(i,threshVal,inequal,weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump,minError,bestClasEst

# D = np.mat(np.ones((5,1))/5)
# dataMat,classLabels = loadSimpData()
# print(bulidStump(dataMat,classLabels,D))

''' 单层决策树的 AdaoostB训练过程 '''
'''
 迭代次数也就是代表的是 基学习器的个数，该算法表示：
 每次我们得到 该分类器的分类权值alpha然后乘以它分类的情况
            下一个分类器 划分权重D 根据 上一个分类器分错而增加，每次给予越大的划分权重，那么就增大了概率 变号（更加朝着反方先去）
            知道所有的情况都分对了 errorRate = 0 ，或 迭代次数达到了 退出
'''

def adaBoostTrainDS(dataArr,classLabels,numIt = 40):                    #numIt 迭代次数，是用户唯一需要指定的参数
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = bulidStump(dataArr,classLabels,D)       #classEst 错误（不相同和标签） 为 1
        # print('D:',D.T)
        alpha = float(0.5 * np.log((1.0-error)/ max(error,1e-16)))          #某一个分类器的权值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst:',classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)       #被分错 则 classLabels[i] 和classEst[i]互为异号 加 -1 结果为正 反之为负
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst                                      #错误率累加计算
        # print('aggClassEst:',aggClassEst.T)
        '''np.sign 返回 数组中元素的符号'''
        aggErrors = np.multiply( np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)) )
        errorRate = aggErrors.sum() / m
        # print('total error: ',errorRate,'\n')
        if errorRate == 0.0 : break
    return weakClassArr,aggClassEst


# classifierArray = adaBoostTrainDS(dataMat,classLabels,9)
# print(classifierArray)

'''
    AdaBoost 分类
'''
def adaClassify(dataToclass,classifierArr):
    dataMatrix = np.mat(dataToclass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

# print(adaClassify([0,0],classifierArray))

'''
    进行 数据分类
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat -1) :
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

def test():
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(dataArr,labelArr,10)

    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testArr,classifierArray)
    m = len(prediction)
    errArr = np.mat(np.ones((m,1)))
    return 1.0-float(errArr[prediction!=np.mat(testLabelArr).T].sum()) / m                      #正确率

'''
    ROC 曲线的绘制 以及 AUC 计算函数
'''

def plotROC(predStrengths,classLabels):
    print(predStrengths)
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:                                    #预测为 正1 的 概率越大排在越前，所以 就是 预测为+1列  当预测为 1 实际为-1 的可能性全在起那面出现，所以更容易从 1 -》 0
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0] - delX],[cur[1],cur[1]- delY],c='b')         #
        cur = (cur[0] - delX,cur[1] - delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('Flase Postive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area Under the Curve is :',ySum*xStep)

dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClassEst.T,labelArr)


