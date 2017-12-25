
# coding: utf-8

# In[ ]:

#LogisticRegression-Imports & Useful Functions
import math
import numpy
def sigmoid(x):
     return 1 / (1 + math.exp(-x))
    
def dLogLikelihood(data, params, i, j, vecLen):
    return data[i][j]*(data[i][vecLen] - sigmoid(numpy.dot(params, data[i][:vecLen])))

def logLikelihood(data, params, vecLen, numVecs):
    total = 0
    for i in range(numVecs):
        y = data[i][vecLen]
        dot = numpy.dot(params, data[i][:vecLen])
        total += (y*math.log(sigmoid(dot)) + (1-y)*(math.log(1 - sigmoid(dot))))
    return total


# In[ ]:

#LogisticRegression-Reformat training data
trainData = []
thetas = []
info = 0
with open('heart-train.txt') as txtFile: #change file as needed on this line
    for line in txtFile:
        if info == 0: 
            trainVecLen = int(line) + 1
            info += 1
            continue
        elif info == 1:
            trainNumVecs = int(line)
            info += 1
            continue
        vec = str.split(line)
        vec = [1] + vec
        vec[trainVecLen-1] = vec[trainVecLen-1].replace(":", "")
        vec = [int(x) for x in vec]
        trainData.append(vec)
theta = [0] * trainVecLen


# In[ ]:

#LogisticRegression-Train
n = .0001
for k in range(10000):
    gradient = [0] * trainVecLen
    for j in range(trainVecLen):
        for i in range(trainNumVecs):
            gradient[j] += dLogLikelihood(trainData, theta, i, j, trainVecLen)
        theta[j] += n*gradient[j]


# In[ ]:

#LogisticRegression-Parameter weights
print theta


# In[ ]:

#2ii,iii-NETFLIX ONLY
zeroes = [0] * trainVecLen
print logLikelihood(trainData, zeroes, trainVecLen, trainNumVecs)
print logLikelihood(trainData, theta, trainVecLen, trainNumVecs)


# In[ ]:

#LogisticRegression-Reformat testing data
testData = []
info = 0
numClass = [0, 0]
with open('heart-test.txt') as txtFile: #change file as needed on this line
    for line in txtFile:
        if info == 0: 
            testVecLen = int(line) + 1
            info += 1
            continue
        elif info == 1:
            testNumVecs = int(line)
            info += 1
            continue
        vec = str.split(line)
        vec = [1] + vec
        vec[testVecLen-1] = vec[testVecLen-1].replace(":", "")
        vec = [int(x) for x in vec]
        testData.append(vec)
        if vec[testVecLen] == 0: 
            numClass[0] += 1
        else: numClass[1] += 1


# In[ ]:

LogisticRegression-Results
correct0 = 0
correct1 = 0
for i in range(testNumVecs):
    likelihood1 = sigmoid(numpy.dot(theta, testData[i][:testVecLen]))
    likelihood0 = 1 - likelihood1
    if likelihood0 > likelihood1 and testData[i][testVecLen] == 0: correct0 += 1
    if likelihood1 > likelihood0 and testData[i][testVecLen] == 1: correct1 += 1
print "Class 0: tested", numClass[0], "correctly classified", correct0
print "Class 1: tested", numClass[1], "correctly classified", correct1
print "Overall: tested", (numClass[0] + numClass[1]), "correctly classified", (correct0 + correct1)
print "Accuracy = ", (float(correct0 + correct1)/float(numClass[0] + numClass[1]))


# In[ ]:

#2d-HEART ONLY (cont.)
accuracies = {}
for m in range(10):
    n = .00002 / (5**m)
    theta = [0] * trainVecLen
    for k in range(10000):
        gradient = [0] * trainVecLen
        for j in range(trainVecLen):
            for i in range(trainNumVecs):
                gradient[j] += dLogLikelihood(trainData, theta, i, j, trainVecLen)
            theta[j] += n*gradient[j]
    correct0 = 0
    correct1 = 0
    for i in range(testNumVecs):
        likelihood1 = sigmoid(numpy.dot(theta, testData[i][:testVecLen]))
        likelihood0 = 1 - likelihood1
        if likelihood0 > likelihood1 and testData[i][testVecLen] == 0: correct0 += 1
        if likelihood1 > likelihood0 and testData[i][testVecLen] == 1: correct1 += 1
    accuracies[n] = (float(correct0 + correct1)/float(numClass[0] + numClass[1]))
print accuracies


# In[ ]:



