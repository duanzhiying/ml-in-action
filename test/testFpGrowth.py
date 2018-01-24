# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:21:54 2018

@author: duanzhiying
"""

import fpGrowth.fpGrowth as fpGrowth

rootNode = fpGrowth.treeNode('pyramid',9,None)
rootNode.children['eye'] = fpGrowth.treeNode('eye',13,None)
data = fpGrowth.loadSimpDat()
initData = fpGrowth.createInitSet(data)
retTree,headerTable = fpGrowth.createTree(initData,3)

value = fpGrowth.findPrefixPath('x',headerTable['x'][1])
freqItems = {}
fpGrowth.mineTree(retTree,headerTable,3,set([]),freqItems)