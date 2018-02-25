# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:48:17 2018

@author: duanzhiying
python代码修改项可参考：http://blog.csdn.net/sinat_17196995/article/details/69621687
"""

import numpy as np

def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

""" 集合切分
    输入参数：数据集合，待切分的特征，该特征的某个值
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

""" 生成叶子节点
"""
def regLeaf(dataSet):#returns the value used for each leaf
    return np.mean(dataSet[:,-1])

""" 计算误差:均方差 * 数量
"""
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

""" 构建树
    输入参数：数据集和可选参数：leafType建立叶子节点的函数，errType误差计算函数，ops其他参数
"""
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

""" 回归树构建的核心函数：用最佳方式（二元切分）切分数据集 + 生成相应的叶节点
    输入参数：同createTree函数
    如下代码的退出情况其实是一种预剪枝（prepruning）
    算法对于tolS和tolN非常敏感
"""
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # tolS和tolN控制函数的停止时机，参数由用户设置。
    # tolS是容许的误差下降值，tolN是切分的最少样本点
    tolS = ops[0]; tolN = ops[1]
    # 如果目标变量相同则退出并返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet) # 退出情况1
    m,n = np.shape(dataSet)
    # S记录误差
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # python3修改
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分后的误差减少不大，则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) # 退出情况2 
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 退出情况3
        return None, leafType(dataSet)
    # 返回最佳切分特征下标，和最佳切分值
    return bestIndex, bestValue

""" 测试输入变量是否是一棵树，返回布尔类型
"""
def isTree(obj):
    return (type(obj).__name__=='dict')

""" 从上到下遍历树直到叶子节点为止，即该函数对树进行塌陷处理（返回树的平均值）
"""
def getMean(tree):
    if isTree(tree['right']): 
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
 
""" 树后剪枝过程
    输入参数：待剪枝的树，剪枝所需的测试数据
"""
def prune(tree, testData):
    # 没有测试数据则坍塌该树
    if np.shape(testData)[0] == 0: return getMean(tree) 
    # 若是子树，则用prune进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): 
        tree['right'] =  prune(tree['right'], rSet)
        
    # 如果都是叶子节点，则进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        # 如果合并后误差小于不合并情况下的误差，则进行合并
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
 
""" 与regLeaf函数类似，负责生成叶节点的模型
    返回回归系数
"""
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

""" 计算线性模型拟合曲线的误差
"""
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

""" 根据dataSet构建X，Y，并构建线性模型，生成回归系数w
    该函数被modelLeaf和modelErr调用
"""
def linearSolve(dataSet):  
    m,n = np.shape(dataSet)
    # 如下处理构建X和Y,且X的第一列为1
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    # 确保在行列式不为0的前提下计算X的转置矩阵
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#######如下代码是对比测试效果使用#######
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat