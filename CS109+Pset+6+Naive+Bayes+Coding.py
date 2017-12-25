
# coding: utf-8

# In[1]:

#My code was created in Jupyter notebook
#For any .txt file, just replace the filenname appropriately in the code "Reformat" chunk before running


# In[2]:

#NaiveBayes-Reformat training data
trainData = []
info = 0
numClass = [0, 0]
with open('netflix-train.txt') as txtFile: #change file as needed on this line
    for line in txtFile:
        if info == 0: 
            trainVecLen = int(line)
            info += 1
            continue
        elif info == 1:
            trainNumVecs = int(line)
            info += 1
            continue
        vec = str.split(line)
        vec[trainVecLen - 1] = vec[trainVecLen-1].replace(":", "")
        vec = [int(x) for x in vec]
        if vec[trainVecLen] == 0: 
            numClass[0] += 1
        else: numClass[1] += 1
        trainData.append(vec)


# In[3]:

#NaiveBayes-Train
pY = [0, 0]
MLE_pXY = {}
LAP_pXY = {}
for j in range(trainVecLen):
    count00 = 0
    count01 = 0
    count10 = 0
    count11 = 0
    key = 'X' + str(j+1)
    for i in range(len(trainData)):
            if trainData[i][j] == 0 and trainData[i][trainVecLen] == 0:
                count00 += 1
            elif trainData[i][j] == 0 and trainData[i][trainVecLen] == 1:
                count01 += 1
            elif trainData[i][j] == 1 and trainData[i][trainVecLen] == 0:
                count10 += 1
            elif trainData[i][j] == 1 and trainData[i][trainVecLen] == 1:
                    count11 += 1
    MLE_pXY[key] = {'00': float(count00)/float(numClass[0]), '01': float(count01)/float(numClass[1]), 
                '10': float(count10)/float(numClass[0]), '11': float(count11)/float(numClass[1])}
    LAP_pXY[key] = {'00': float(count00 + 1)/float(numClass[0] + 2), '01': float(count01 + 1)/float(numClass[1] + 2), 
                '10': float(count10 + 1)/float(numClass[0] + 2), '11': float(count11 + 1)/float(numClass[1] + 2)}
pY[0] = float(numClass[0])/float(trainNumVecs)
pY[1] = float(numClass[1])/float(trainNumVecs)


# In[4]:

#NaiveBayes-Reformat testing data
testData = []
info = 0
numClass = [0, 0]
with open('netflix-test.txt') as txtFile: #change file as needed on this line
    for line in txtFile:
        if info == 0: 
            testVecLen = int(line)
            info += 1
            continue
        elif info == 1:
            testNumVecs = int(line)
            info += 1
            continue
        vec = str.split(line)
        vec[testVecLen - 1] = vec[testVecLen-1].replace(":", "")
        vec = [int(x) for x in vec]
        if vec[testVecLen] == 0: 
            numClass[0] += 1
        else: numClass[1] += 1
        testData.append(vec)


# In[5]:

#NaiveBayes-Test
MLE_correct0 = 0
MLE_correct1 = 0
LAP_correct0 = 0
LAP_correct1 = 0
likelyList = []
error = []
for i in range(len(testData)):
    MLE_likelihood0 = 1
    MLE_likelihood1 = 1
    LAP_likelihood0 = 1
    LAP_likelihood1 = 1
    for j in range(testVecLen):
        key = 'X' + str(j+1)
        if testData[i][j] == 0:
            MLE_likelihood0 *= MLE_pXY[key]['00']
            MLE_likelihood1 *= MLE_pXY[key]['01']
            LAP_likelihood0 *= LAP_pXY[key]['00']
            LAP_likelihood1 *= LAP_pXY[key]['01']
        elif testData[i][j] == 1:
            MLE_likelihood0 *= MLE_pXY[key]['10']
            MLE_likelihood1 *= MLE_pXY[key]['11']
            LAP_likelihood0 *= LAP_pXY[key]['10']
            LAP_likelihood1 *= LAP_pXY[key]['11']
    if MLE_likelihood0*pY[0] > MLE_likelihood1*pY[1] and testData[i][testVecLen] == 0: MLE_correct0 += 1
    if MLE_likelihood1*pY[1] > MLE_likelihood0*pY[0] and testData[i][testVecLen] == 1: MLE_correct1 += 1
    if LAP_likelihood0*pY[0] > LAP_likelihood1*pY[1] and testData[i][testVecLen] == 0: LAP_correct0 += 1
    if LAP_likelihood1*pY[1] > LAP_likelihood0*pY[0] and testData[i][testVecLen] == 1: LAP_correct1 += 1
    #Looking for mistake prediction
    if not error and MLE_likelihood0*pY[0] > MLE_likelihood1*pY[1] and testData[i][testVecLen] == 1: 
        error = testData[i]


# In[6]:

#2i FOR NETFLIX ONLY
print pY[1]
#2ii,iii FOR NETFLIX ONLY
for n in range(len(MLE_pXY)):
    key = 'X' + str(n+1)
    print n+1, MLE_pXY[key]['11'], MLE_pXY[key]['10'], LAP_pXY[key]['11'], LAP_pXY[key]['10']


# In[7]:

#3i FOR NETFLIX ONLY
for n in range(len(MLE_pXY)):
    key = 'X' + str(n+1)
    numerator = (MLE_pXY[key]['11']*pY[1])/(MLE_pXY[key]['11']*pY[1] + MLE_pXY[key]['10']*pY[0]) #P(Y=1|X=1) = P(X=1|Y=1)P(Y=1)/P(X=1)
    denominator = (MLE_pXY[key]['01']*pY[1])/(MLE_pXY[key]['01']*pY[0] + MLE_pXY[key]['00']*pY[1]) #P(Y=1|X=0) = P(X=0|Y=1)P(Y=1)/P(X=0)
    print n+1, numerator/denominator
#3ii FOR NETFLIX ONLY
print error
print MLE_pXY['X19']['01'], MLE_pXY['X5']['01'], MLE_pXY['X18']['01']


# In[8]:

#NaiveBayes-MLE Results
print ('\033[1m{:10s}\033[0m'.format('M L E'))
print "Class 0: tested", numClass[0], "correctly classified", MLE_correct0
print "Class 1: tested", numClass[1], "correctly classified", MLE_correct1
print "Overall: tested", (numClass[0] + numClass[1]), "correctly classified", (MLE_correct0 + MLE_correct1)
print "Accuracy = ", (float(MLE_correct0 + MLE_correct1)/float(numClass[0] + numClass[1]))


# In[9]:

#NaiveBayes-Laplace Results
print ('\033[1m{:10s}\033[0m'.format('Laplace'))
print "Class 0: tested", numClass[0], "correctly classified", LAP_correct0
print "Class 1: tested", numClass[1], "correctly classified", LAP_correct1
print "Overall: tested", (numClass[0] + numClass[1]), "correctly classified", (LAP_correct0 + LAP_correct1)
print "Accuracy = ", (float(LAP_correct0 + LAP_correct1)/float(numClass[0] + numClass[1]))


# In[ ]:



