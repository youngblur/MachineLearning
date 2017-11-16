#coding=utf-8

import matplotlib.pyplot as plt
import matplotlib
from Classification.ID3 import ID3


decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.axl.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',xytext = centerPt, textcoords = 'axes fraction',va="center",ha="center",bbox = nodeType,arrowprops = arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    createPlot.axl = plt.subplot(111,frameon = False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

createPlot()
# print(matplotlib.matplotlib_fname())
'''
    求myTree 的 叶节点的个数 和 最大的深度 
    叶节点个数 ：所有的个数，对每一个递归都需要累加
    最大深度 ： 一旦到达叶子节点则返回，并将深度 + 1
'''


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]                       #在python2.x中，dict.keys()返回一个列表，在python3.x中，dict.keys()返回一个dict_keys对象，比起列表，这个对象的行为更像是set，所以不支持索引的。解决方案：list(dict.keys())[index]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth =  0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if maxDepth < thisDepth:
            maxDepth = thisDepth
    return maxDepth

dataSet,labels = ID3.createDataSet()
myTree = ID3.createTree(dataSet,labels)
print(getNumLeafs(myTree))
print(getTreeDepth(myTree))
