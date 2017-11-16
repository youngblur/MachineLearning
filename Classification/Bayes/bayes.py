#encoding = utf-8

'''
    利用 Bayes 进行文本分类
'''
import numpy as np
import math

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 is not
    return postingList,classVec


'''
    将词 进行合并 去掉重复的 得到特征
'''
def createVocalbList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)         # 取 并集
    return list(vocabSet)


'''
    将一行的数据集 转化为 根据总特征（词汇表）的 向量 词集模型（词是否出现 0，1）
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word : %s is not in my Vocabulary!" % word)
    return returnVec


'''
    词袋模型（词 出现的次数）
'''
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word : %s is not in my Vocabulary!" % word)
    return returnVec


# 对我们的 数据集 构造 单词 表序列

#dataset,classVec = loadDataSet()
# vocabulary = createVocalbList(dataset)
# print(vocabulary)

'''
    trainMatrix ： 训练集 也就是（转化为词汇表的 向量）
    trainCategory: 判断是否为 侮辱词汇 也就是（classVec）
    返回 ： 在 词汇属性为0 （不侮辱）的情况下 个特征的概率 列表 
            在         1     侮辱         
            1（侮辱） 占的 比例
'''
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p_abusive = sum(trainCategory) / float(len(trainCategory))

    # p0_num = np.zeros(numWords)
    # p1_num = np.zeros(numWords)
    # p0_all = 0.0
    # p1_all = 0.0
    #   排除概率等于零 所以 给每一个特征都给一个初始值1
    p0_num = np.ones(numWords)
    p1_num = np.ones(numWords)
    p0_all = numWords
    p1_all = numWords

    for line in range(numTrainDocs):
        if trainCategory[line] ==  1:                #1 表示 是侮辱词汇
            p1_num += trainMatrix[line]             #p1_num 表示各特征出现的次数
            p1_all += sum(trainMatrix[line])         #p1_all 表示 词汇为1 时， 的所有特征出现的次数
        else:
            p0_num += trainMatrix[line]
            p0_all += sum(trainMatrix[line])

    # p1Vect = p1_num/p1_all                          # p1_num 里每一个 特征（单词） 的次数 除以 总数 p1_all = sum(p1_num）
    # p0Vecc = p0_num/p0_all
    # 考虑到 概率相乘到一个比较小的之后，计算机默认为零 所以扩大 这个概率 用log 函数
    p1Vect = [math.log(num/p1_all) for num in p1_num]
    p0Vect = [math.log(num/p0_all) for num in p0_num]   # math.log() 不能直接对矩阵用
    return p0Vect,p1Vect,p_abusive

# dataset,classVec = loadDataSet()
# vocabulary = createVocalbList(dataset)
# print(vocabulary)
# trainMat = []
# for line in dataset:
#     trainMat.append(setOfWords2Vec(vocabulary,line))
# p0_v,p1_v,p_abusive = trainNB(trainMat,classVec)
# print(p0_v)
# print(p_abusive)

'''
    vect2Classify: 准备去分类的向量  [0，1，0，1，。。。。。。] 长度和 vacabulary一样，所以 1 代表 有该特征， 0 代表 无
    p0Vect: 每个特征存储的在 0 的条件下 的概率  的log的向量
    p1Vect：每个特征存储的在 1 的条件下 的概率  的log的向量   所以 用sum 和之后 （+）  在logA+logB = logA*B
    pClassIs1 ：代表 标签向量 是 1 的 概率 
'''
def classifyNB(vect2Classify,p0Vect,p1Vect,pClassIs1):
    p1 = sum([x*y for x,y in zip(vect2Classify,p1Vect)]) + math.log(pClassIs1)              # 对两个list相乘 vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    p0 = sum([x*y for x,y in zip(vect2Classify,p0Vect)]) + math.log(1-pClassIs1)
    if(p1>p0):
        return 1
    else:
        return 0

def testingNB():
    listData,listClasses = loadDataSet()
    myVocabulary = createVocalbList(listData)
    trainMat = []
    for line in listData:
        trainMat.append(setOfWords2Vec(myVocabulary,line))
    p0V,p1V,pAbusive = trainNB(trainMat,listClasses)

    testList = ['love','my','dalmation']
    testVec = setOfWords2Vec(myVocabulary,testList)
    print(str(testList)+'  is classified as : ',classifyNB(testVec,p0V,p1V,pAbusive))
    testList2 = ['stupid','garbage']
    testVec2 = setOfWords2Vec(myVocabulary,testList2)
    print(str(testList2)+'  is classified as : ',classifyNB(testVec2,p0V,p1V,pAbusive))

# testingNB()
'''
    输入文本的内容，得到该文本的 所有分词 并删除掉（长度小于2 的分词）
'''
def textParse(bigString):
    import re
    listOfTokens = re.split(r"\W*",bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    import random
    docList = [];   classList = [] ;    fullTest = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabulary = createVocalbList(docList)
    trainingSet = list(range(50))
    testSet = []

    #10个做测试 ，40 个做训练
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = [];  trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabulary,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB(trainMat,trainClass)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabulary,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is : ",float(errorCount)/len(testSet))

spamTest()



