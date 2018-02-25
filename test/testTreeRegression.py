# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:47:28 2018

@author: duanzhiying
"""
import numpy as np
import regression.treeRegression as tr
#testMat = np.mat(np.eye(4))
#mat0,mat1 = tr.binSplitDataSet(testMat, 1, 0.5)

#myDat = tr.loadDataSet('../data/ex00.txt')
#myMat = np.mat(myDat)
#tree = tr.createTree(myMat)

myDat2 = tr.loadDataSet('../data/ex2.txt')
myMat2 = np.mat(myDat2)
myTree = tr.createTree(myMat2, ops=(0,1))
myDatTest = tr.loadDataSet('../data/ex2test.txt')
myMat2Test = np.mat(myDatTest)
tree = tr.prune(myTree, myMat2Test)