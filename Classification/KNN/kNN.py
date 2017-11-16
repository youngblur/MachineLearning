#encoding = utf-8

'''
    datingTestSet.txt  数据文本
    datingTestSet2.txt large -> 3 small ->2  didnt ->1
'''
import numpy as np

# 构造 KNN 分类器
'''
    二维向量的 进行分类
    inX:        用于分类的输入向量
    dataset:    输入的训练样本集 不包含最后的标签向量
    lables:     标签向量
    k:          选择最近邻居的数目
'''
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]          #取 dataSet 的行数（数据量）
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet    #将 inX 变成 行数 dataSetSize 列数：1 个 inX
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()    #得到排序后的序号
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1
    sortedClassCount = sorted(classCount.items(),key = lambda item:item[1],reverse = True)
    return sortedClassCount[0][0]

'''
    将文本记录记录转换为 NumPy 的解析程序
    input：文件路径
    output： 向量矩阵 和 标签向量
'''
def file2maxtrix(filename):
    fr = open(filename)
    arrayOLines  = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))   # 得到一个 numberOfLines 行 3 列的零向量
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()     #去掉开头和结尾的空格
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# datingDataMat,datingLabels = file2maxtrix("datingTestSet2.txt")               #ScatterPlot 时 需要
# print(datingDateMat)
# print(datingLabels[0:20])

'''
    归一化：使每一列数据（每一个特征） 都在相同的取值范围内，这样在计算距离时，它们的权衡是相等的
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)        # 取每一列的最小值 构成的矩阵
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

'''
    测试算法
'''
def datingClassTest():
    hoRatio = 0.10          # 前 10% 当做输入向量（进行测试的）  后 90% 当做训练向量（数据集）
    datingDataMat,datingLabels = file2maxtrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)    #测试数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with :%d, the real answer is : %d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f " % (errorCount / float(numTestVecs)))

'''
    使用算法构建完整的系统，用户输入 三个参数的 数值 利用kNN算法 得到label
'''
def classifyPerson():
    resultLisy = ["not in all", "in small doses", "in large doses"]

    percentTats = float(input("percentage of time spent playing video game?"))
    ffMiles =float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat,datingLabels = file2maxtrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person :", resultLisy[classifierResult-1])


if  __name__ == "__main__":
     datingClassTest()
#     classifyPerson()
