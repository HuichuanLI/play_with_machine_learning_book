import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def meanX(data):
    return np.mean(data, axis=0)
def compute_si(xi):
    n = xi.shape[0]
    ui = meanX(xi)
    si = 0
    for i in range(0, n):
        si = si + (xi[i, :] - ui).T * (xi[i, :] - ui)
    return si
def compute_Sb(x1, x2):
    dataX=np.vstack((x1,x2))
    print ("dataX:", dataX)
    u1=meanX(x1)
    u2=meanX(x2)
    u=meanX(dataX)
    Sb = (u-u1).T * (u-u1) + (u-u2).T * (u-u2)
    return Sb
def LDA(x1, x2):
    s1 = compute_si(x1)
    s2 = compute_si(x2)
    Sw = s1 + s2
    Sb = compute_Sb(x1, x2)
    eig_value, vec = np.linalg.eig(np.mat(Sw).I * Sb)
    index_vec = np.argsort(-eig_value)
    eig_index = index_vec[:1]
    w = vec[:, eig_index]
    return w
def createDataSet():
    X1 = np.mat(np.random.random((8, 2)) * 5 + 15)
    X2 = np.mat(np.random.random((8, 2)) * 5 + 2)
    return X1, X2
def plotFig(group):
    fig = plt.figure()
    plt.ylim(0, 30)
    plt.xlim(0, 30)
    ax = fig.add_subplot(111)
    ax.scatter(group[0,:].tolist(), group[1,:].tolist())
    plt.show()
if __name__ == "__main__":
    x1, x2 = createDataSet()
    print(x1, x2)
    w = LDA(x1, x2)
    print ("w:", w)
plotFig(np.hstack((x1.T, x2.T)))