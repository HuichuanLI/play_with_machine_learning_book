import operator
def createDateset():
    group = [[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]]
    labels = ['A','A','B','B']
    return group, labels
def kNNClassify(newInput, dataSet,labels,k):
    temp = [[0,0],[0,0],[0,0],[0,0]]
    i = 0
    for i in range(4):
        j=0
        for j in range(2):
            temp[i][j] = dataSet[i][j] - newInput[j]
    i = 0
    for i in range(4):
        j = 0
        for j in range(2):
            temp[i][j] = temp[i][j] ** 2
        distance = [0,0,0,0]
    i = 0
    for i in range(4):
        j=0
        for j in range(2):
            distance[i] = distance[i] +temp[i][j]
    i = 0
    for i in range(4):
        distance[i] = distance[i] ** 0.5
    sortDistance = [0,1,2,3]
    i=0
    for i in range(4):
        j = 0
        for j in range(3):
            if distance[j]>distance[j+1]:
                temp = distance[j]
                distance[j] = distance[j+1]
                distance[j+1] = temp
            temp = sortDistance[j]
    sortDistance[j] = sortDistance[j+1]
    sortDistance[j + 1] = temp
    votelabels = [" ", " ", " ", " "]
    i = 0
    for i in range(k):
        if i < k:
            votelabels[i] = labels[sortDistance[i]]
    countA = 0
    countB = 0
    i = 0
    for i in range(k):
        if votelabels[i] == 'A':
            countA = countA + 1
        else:
            countB = countB + 1
        if countA > countB:
            maxLabel = 'A'
        else:
            maxLabel = 'B'
    return maxLabel


def test():
    dataSet, labels = createDateset()
    testX = [1.2, 1.0]
    k = 3
    outputLabel = kNNClassify(testX, dataSet, labels, k)
    print("Your input is:", testX, "and classified to class: ", outputLabel)
    testX = [0.1, 0.3]
    outputLabel = kNNClassify(testX, dataSet, labels, k)
    print("Your input is:", testX, "and classified to class: ", outputLabel)
test()
