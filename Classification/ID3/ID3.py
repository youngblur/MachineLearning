#encoding = utf-8

'''
    ID3 算法 处理离散特征
    计算每一列（每个特征）的 熵（香农熵，熵越大，则混合的数据越多） 值，；
    再计算 信息增益 ，选择 信息增益最大的去划分
    每一次划分都会消耗 特征（剔除特征）
'''

import numpy as np
from math import log

'''
    不浮出水面是否可以生存             是否有脚蹼               属于鱼类
            是                           是                      是
            是                           是                      是
            是                           否                      否
            否                           是                      否
            否                           是                      否
'''
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no'] ]
    labels = ['can live without surfacing','has it flippers']
    return dataSet,labels

'''
    计算给定数据集的 标签向量 的 熵
'''
def calEntropy(dataSet):
    num = len(dataSet)
    labelCount = {}
    for featVec in  dataSet:
        currentLabel = featVec[-1]
        labelCount[currentLabel]  = labelCount.get(currentLabel,0) + 1          #统计 标签的 各类型 的个数
    entropy = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/num
        entropy -= prob*log(prob,2)
    return  entropy


'''
    按照给定的特征划分数据集（得到的子数据集是满足 进行划分的特征值 是 == value 的）
    dataSet: 数据集
    axis：   需要进行划分的特征（列号）
    value：  划分的 特征值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:                          # 抽取 第 axis 列 每一行 等于 value  时 的 数据 存起来 （）
            reducdFeatVec = featVec[:axis]
            reducdFeatVec.extend(featVec[axis+1:])            #然后 剔除 该行后 得到 子数据
            retDataSet.append(reducdFeatVec)
    return retDataSet

'''
    关于  extend  和   append
    a = [1,2,3]
    b = [4,5,6]
    a.append(b)    --->    [1,2,3,[4,5,6]]
    a.extend(b)    --->    [1,2,3,4,5,6]
'''

'''
    选择最好的数据集划分方式
'''

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)               # 得到标签向量的 熵值
    bestInfoGain = 0.0
    bestFeature = -1                                #选择 最好的特征（列）
    for i in range(numFeatures):
        featureList = [rows[i] for rows in dataSet]
        uniqueVal = set(featureList)
        newEntropy = 0.0
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = float(len(subDataSet)) / float(len(dataSet))
            newEntropy += prob*calEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
    如果数据集已经处理了所有的属性，但是类标签依然不是唯一的，此时我们采用多数表决的方法决定该叶子节点的分类
    既 只剩下 类标签向量 ---classList 里面的量不唯一
'''
def majorityCnt(classList):
    classCount = {}
    for _class in classList:
        classCount[_class] = classCount.get(_class,0) + 1
    sortedClassCount = sorted(classCount.items(),key = lambda item :item[1], reverse=True)
    return sortedClassCount[0][0]

'''
    创建树的函数代码(递归)
'''
def createTree(dataSet,_labels):
    labels = _labels[:]
    classList = [rows[-1] for rows in dataSet]
    if classList.count(classList[0]) == len(classList):                 #如果剩下的类标签唯一 停止划分 返回该值
        return classList[0]
    if len(dataSet[0]) == 1:                                            #已经便利完所有特征（只剩下 标签向量） 则返回出现次数最多的值
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featureValues = [rows[bestFeature] for rows in dataSet]             #最佳划分 特征的 所有特征值
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree

# dataSet,labels = createDataSet()
# tree = createTree(dataSet,labels)
# print(tree)

'''
    使用决策树的分类函数
    inputTree：利用 dict 保存的 决策树
    featureLabels: 特征标签 也就是 该特征的含义 
    testVec: 不包含标签的向量
'''
def classify(inputTree,featureLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ ==  'dict':
                classLabel = classify(secondDict[key],featureLabels,testVec)
            else:
                classLabel = secondDict[key]

    return classLabel

'''
    使用 pickle 模块 来存储和读取数据
    pickle 模块 可以进行结构化存储和读取 也就是（可以存 dict ， 读也是dict）
    而普通的 write 和 read 会破坏结构 变为字符串 再进行操作
'''
def stroeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def readTree(filename):
    import pickle
    fr = open(filename,'r')
    return pickle.load(fr)



#
# dataSet,labels = createDataSet()
# myTree = createTree(dataSet,labels)
# label = classify(myTree,labels,[1,0])
# print(label)



