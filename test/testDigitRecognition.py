from kNN import kNN
from kNN import digitRecognition

def handwritingClassTest():
    m,trainingMat, hwLabels =  digitRecognition.dataSetBuild('../data/trainingDigits')
    m_test,testMat,hwLabels_test = digitRecognition.dataSetBuild('../data/testDigits')
    errorCount = 0.0
    for i in range(m_test):
        classifierResult = kNN.classify0(testMat[i,:],trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, hwLabels_test[i]))
        if(classifierResult != hwLabels_test[i]):
            errorCount += 1.0
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount/float(m_test)))
    
handwritingClassTest()