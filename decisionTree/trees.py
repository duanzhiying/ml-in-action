# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:52:41 2018

@author: duanzhiying
"""

"""创建数据集
"""
from decisionTree import *

def createDataset():
    dataSet = np.array([[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']])
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

"""计算分类的信息熵
"""
def calcShannonEnt(dataSet):
    numEntries = dataSet.shape[0]
    labelCounts = {}
    # 统计各分类的信息数
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 计算分类的总的信息熵entropy
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) # log base 2
    # 熵越高，则混合的数据也越多
    return shannonEnt

"""按照axis特征的value划分数据集，得到不包括该特征的数据信息
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featureVect in dataSet:
        if featureVect[axis] == value:
            reducedFectVect = list(featureVect[:axis])
            reducedFectVect.extend(list(featureVect[axis+1:]))
            retDataSet.append(reducedFectVect)
    return np.array(retDataSet)

"""选择最好的数据划分方式，即确定特征
"""
def chooseBestFeatureToSplit(dataSet):
    # 获取特征总数
    numOfFeature = dataSet.shape[1] - 1
    bestInfoGain = 0
    bestFeature = -1
    baseEnt = calcShannonEnt(dataSet)
    for i in range(numOfFeature):
        featureValueList = [feature[i] for feature in dataSet]
        uniqueFeatureValue = set(featureValueList)
        splitEnt = 0
        for value in uniqueFeatureValue:
            splitData = splitDataSet(dataSet, i, value)
            # 计算该特征值出现的概率
            prob = len(featureValueList)/float(dataSet.shape[0])
            splitEnt += prob * calcShannonEnt(splitData)
            # 计算信息增益
        infoGain = baseEnt - splitEnt
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature, bestInfoGain
    
"""多数表决，获取分类
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key = operator.itemgetter(1), 
                              reverse = True)
    return sortedClassCount[0][0]

"""构造决策树
"""
def createTree(dataSet, labels):
    useLabels = copy.deepcopy(labels)
    classList = [value[-1] for value in dataSet]
    # 当仅有一个类别时，返回该类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果只有类别，没有特征，则按照多数表决确定类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeature, infoGain = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = useLabels[bestFeature]
    
    myTree = {bestFeatureLabel : {}}
    del(useLabels[bestFeature])
    featureValues = [value[bestFeature] for value in dataSet]
    uniqueFeatureValues = set(featureValues)
    
    for value in uniqueFeatureValues:
        subLabels = useLabels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value),
            subLabels)
    return myTree

"""使用决策树进行分类测试
"""
def classify(inputTree, featLabels, testVect):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 获取第一个特征的index
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVect[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVect)
            else:
                classLabel = secondDict[key]
    return classLabel
    
dataSet,labels = createDataset()
#shannonEnt = calcShannonEnt(dataSet)
#retDataSet = splitDataSet(dataSet, 0, '0')
#bestFeature, bestInfoGain = chooseBestFeatureToSplit(dataSet)
myTree = createTree(dataSet, labels)
cls = classify(myTree, labels, ['1','1'])
