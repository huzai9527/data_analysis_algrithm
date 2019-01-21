# _*_ coding:utf-8 _*_

import numpy as np

# 训练集
def loadDataSet():
    """

    导入数据， 1代表脏话

    @ return postingList: 数据集

    @ return classVec: 分类向量

    """
    # 这里的训练集是通过jieba等工具分割的，必须是每个因子，而不是一句话
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],

                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],

                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],

                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],

                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],

                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec

# 构造词库
def createVocabList(dataSet):
    """

    创建词库

    @ param dataSet: 数据集

    @ return vocabSet: 词库

    """

    vocabSet = set([])

    for document in dataSet:
        # 求并集

        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """

    文本词向量.词库中每个词当作一个特征，文本中就该词，该词特征就是1，没有就是0

    @ param vocabList: 词表

    @ param inputSet: 输入的数据集

    @ return returnVec: 返回的向量

    """

    returnVec = [0] * len(vocabList)

    for word in inputSet:

        if word in vocabList:

            returnVec[vocabList.index(word)] += 1  #？？？这里是+1 还是直接等于1

        else:

            print("单词: %s 不在词库中!" % word)

    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """

    训练

    @ param trainMatrix: 训练集

    @ param trainCategory: 分类

    """

    numTrainDocs = len(trainMatrix)

    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)  # “滥用”概率

    # 防止某个类别计算出的概率为0，导致最后相乘都为0，所以初始词都赋值1，分母赋值为2.

    p0Num = np.ones(numWords)

    p1Num = np.ones(numWords)

    p0Denom = 2

    p1Denom = 2

    for i in range(numTrainDocs):

        if trainCategory[i] == 1:

            p1Num += trainMatrix[i]      #在分类1中每个特征向量出现的次数

            p1Denom += sum(trainMatrix[i])  # 记录一共出现多少个单词

        else:

            p0Num += trainMatrix[i]

            p0Denom += sum(trainMatrix[i])

    # 这里使用log函数，方便计算，因为最后是比较大小，所有对结果没有影响。

    p1Vect = np.log(p1Num / p1Denom)  # 计算类标签为1时的其它属性发生的条件概率

    p0Vect = np.log(p0Num / p0Denom)  # 计算标签为0时的其它属性发生的条件概率

    return p0Vect, p1Vect, pAbusive  # 返回条件概率和类标签为1的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    判断大小

    """

    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # logp1 + logp3 + logp7 = log(p1*p3*p7) 等同于 p1*p3*p7

    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)

    if p1 > p0:

        return 1

    else:

        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    # 生成词表
    myVocabList = createVocabList(listOPosts)

    trainMat = []
    # 统计每个特征出现的次数
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    #计算每个特正在每个分类中的发生概率
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage', 'bob']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    testingNB()

