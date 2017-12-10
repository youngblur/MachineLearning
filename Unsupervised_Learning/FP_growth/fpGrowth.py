# encoding = utf-8

# FP tree 数据结构
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        '''
        :param nameValue: 名字变量
        :param numOccur: 计数值
        :param parentNode: 父节点
        '''
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    #对 count 变量进行加值
    def inc(self,numOccur):
        self.count += numOccur

    #显示节点
    def disp(self,ind = 1):
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)


'''
    1.扫描一遍事物，统计各个元素的次数
    2.对少于支持度的元素进行删除
    3.事物按照第一次剩下的元素的顺序进行排序（避免相同相不同顺序的重复出现）
    4.构建FP—tree
'''

def createTree(dataSet,minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            if item not in headerTable:
                headerTable[item] = [0,None]
            headerTable[item][0] += 1

    for k in list(headerTable.keys()):
        if headerTable[k][0] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())

    if len(freqItemSet) == 0:
        return None,None

    retTree = treeNode('Null Set',1,None)
    for trans in dataSet:
        localD = {}
        for item in trans:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key = lambda x :(x[1],x[0]),reverse=True)]  #先用数字比较 ，再用 其 字母比较，降序
            updateTree(orderedItems,retTree,headerTable)
    return retTree,headerTable

def updateTree(items,inTree,headerTable):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(1)
    else:
        inTree.children[items[0]] = treeNode(items[0],1,inTree)

        if headerTable[items[0]][1] == None:
            #如果是第一次出现 则让 headerTable 指向它
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            #如果不是第一次出现 则找到 它第一次出现的点（存在headerTable中） 一直找到最后出现
            nextLink = headerTable[items[0]][1]
            while(nextLink.nodeLink != None):
                    nextLink = nextLink.nodeLink
            nextLink.nodeLink =inTree.children[items[0]]

    if len(items) > 1:
        updateTree(items[1:],inTree.children[items[0]],headerTable)


def loadSimpleData():
    simpleData = [['r', 'z', 'h', 'j', 'p'],
                   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                   ['z'],
                   ['r', 'x', 'n', 'o', 's'],
                   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleData


simpleData = loadSimpleData()
myFPtree,myHeaderTab = createTree(simpleData,3)
myFPtree.disp()


'''
    发现以给定元素项结尾的所有路径的函数
'''
def ascendTree(leafNode,nodeWeight,count):
    if leafNode.parent != None:
        nodeWeight[leafNode.name] = nodeWeight.get(leafNode.name,0)+count
        ascendTree(leafNode.parent,nodeWeight,count)

def findFreqNode(treeNode,minSup):
    nodeWeight = {}
    while treeNode !=None:
        ascendTree(treeNode,nodeWeight,treeNode.count)
        treeNode = treeNode.nodeLink

    print(nodeWeight)
    nodeList = []
    for node in nodeWeight.keys():
        if nodeWeight[node] >= minSup:
            nodeList.append(node)

    returnList = []
    for i in range(1,len(nodeList)+1,1):
        import  itertools
        returnList.extend(list(itertools.combinations(nodeList, i)))

    return returnList


# 查找频繁项集
def mineTree(headerTable,minSup):
    bigL = [v[0] for v in sorted(headerTable.items(),key = lambda x :(x[1][0],x[0]))]  #先用数字比较 ，再用 其 字母比较，升序

    freqDict = {}
    for basePat in bigL:
        node = headerTable[basePat][1]
        freqDict[node.name] = findFreqNode(node,minSup)

    freqSet = set()
    for values in freqDict.values():
        for value in values:
            freqSet.add(value)

    print(freqSet)

mineTree(myHeaderTab,3)