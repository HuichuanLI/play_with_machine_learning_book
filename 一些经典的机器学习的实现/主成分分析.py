from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)
def pca(dataMat, topNfeat=999999):
    meanVals = mean(dataMat, axis=0)
    DataAdjust = dataMat - meanVals
    covMat = cov(DataAdjust, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = DataAdjust * redEigVects
    print( "X: ",lowDDataMat )
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat,1)
print( "shape(lowDMat): ",shape(lowDMat))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()