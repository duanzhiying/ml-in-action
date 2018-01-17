# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:11:15 2018

@author: duanzhiying
"""

"""创建数据集
   返回：array
"""
def createDataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


"""创建集合C1,C1是大小为1的所有候选项集的集合。
   之后需要构建C2、C3、...CN
"""
def createC1(dataset):
    C1 = []
    for data in dataset:
        for item in data:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    # 生成不可变集合
    # python3中map是一个iterator对象，只能被遍历一次
    return list(map(frozenset,C1))


"""输入参数：数据集，候选项集，最小支持度
   输出参数：满足最小支持度的项集，项集和支持度构成的字典
"""
def scanD(D, CK, minSupport):
    ssCnt = {}
    retList = []
    supportData = {}
    for data in D:
        for can in CK:
            if can.issubset(data):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 数据集的条数
    numItems = float(len(D))
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

""" 根据LK-1项集生成CK候选项集
"""
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

    
"""apriori算法的主程序
"""
def apriori(dataset, minSupport = 0.5):
    C1 = createC1(dataset)
    L1,supportData = scanD(dataset, C1, minSupport)
    L = [L1]
    k = 2 
    while (len(L[k-2]) > 0):
        CK = aprioriGen(L[k-2],k)
        LK,supK = scanD(dataset, CK, minSupport)
        # 将supK插入到supportData字典中
        supportData.update(supK)
        L.append(LK)
        k += 1
    return L, supportData
        
"""生成关联规则主函数
   输入参数：频繁项集，频繁项集对应的支持度，最小置信度
   输出参数：关联规则
"""
def generateRules(L, supportData, minConf=0.5):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if(i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList,
                                minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    
    return bigRuleList
    
""" 计算频繁项集数大于2时的置信度：生成后件，计算置信度
    问题：为什么对于大于2的项集不计算后件为单个元素的置信度
"""
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.5):
    m = len(H[0])
    # 该判断的目的是根据项集数为m生成m+1项集作为后件，是否符合要求（小于项集长度）
    if (len(freqSet) > (m + 1)):
        # 生成Hm+1的新的候选项集
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 当len大于1，说明至少有2个规则满足条件，则可用于进一步的合并
        if (len(Hmp1) > 1):    
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

"""计算频繁项集数为2时的置信度
"""
def calcConf(freqSet, H, supportData, brl, minConf=0.5):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
            
dataset = createDataset()
L, supportData = apriori(dataset)
bigRuleList = generateRules(L, supportData)
