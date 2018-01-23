# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:18:04 2018

@author: duanzhiying
"""

"""定义存储FP树的数据结构
"""
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # 节点名称
        self.count = numOccur # 节点计数
        self.nodeLink = None # 链接相似元素
        self.parent = parentNode # 父节点，needs to be updated
        self.children = {} # 子节点，是字典类型

    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

"""导入数据集
"""
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

"""将数据格式化为字典类型，key对应数据集的每一行，value为1
"""
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

"""构造FP树
"""
def createTree(dataSet, minSup=1): 
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 删除小于最小支持度的项
    for k in list(headerTable): 
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] 
    # 创造Tree
    retTree = treeNode('Null Set', 1, None) 
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # 按照出现次数倒序排
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

"""输入参数：倒序排的项，FP树，头指针，计数
"""
def updateTree(items, inTree, headerTable, count):
    # 如果该节点已经是FP树的子节点了，则只更新计数
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count) 
    else:   # 如果该FP树还没有该节点，则增加并作为children，且设置parentNode，且将该节点的头指针指向该节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 如果items还有项，则继续处理，且将inTree的起始节点修改为当前节点，且items删掉当前节点
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

""" 更新header，构造nodelink
    header头指针从nodeLink开始，一直沿着nodeLink到达该链表的末尾
"""
def updateHeader(nodeToTest, targetNode):   
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
    
"""追溯到根节点
"""
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
"""查找项basePat的前缀路径
   输入参数：basePat要查找的项，treeNode是该项的起始点（header头指针指向的）
"""
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) >= 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

"""查找频繁项集
   输入参数：freqItemList保存频繁项集列表
"""
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 对头指针的项进行排序，按照count升序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    # 遍历bigL（所有头指针项）
    for basePat in bigL:  
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # 将每个频繁项添加到频繁项集
        freqItemList.append(newFreqSet)
        # 查找basePat的条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 基于条件模式基重新构建FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)  
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)