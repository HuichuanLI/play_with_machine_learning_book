import numpy as np
import matplotlib.pyplot as plt

np.random.seed()
X = np.array([[2, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 2, 3],
              [0, 0, 0, 1],
              [1, 2, 2, 1]])


class LSA:
    def __init__(self, x, k):  # k为话题个数
        self.k = k
        self.X = x  # 单词文本矩阵
        self.m = x.shape[0]  # 单词数
        self.n = x.shape[1]  # 文本数
        self.W = np.random.uniform(0, 1, (self.m, k))  # 初始化单词-话题矩阵
        self.H = np.random.uniform(0, 1, (k, self.n))  # 初始化话题-文本矩阵

    def standard(self):
        # 对每个话题的向量归一化（W列向量归一化）
        t = self.W ** 2
        T = np.sqrt(np.sum(t, axis=0))
        for i in range(self.W.shape[0]):
            self.W[i] = self.W[i] / T

    def update(self):
        # 平方损失的更新公式
        up1 = np.dot(self.X, self.H.T)
        t1 = np.dot(self.W, self.H)
        down1 = np.dot(t1, self.H.T)
        up2 = np.dot(self.W.T, self.X)
        t2 = np.dot(self.W.T, self.W)
        down2 = np.dot(t2, self.H)
        for i in range(self.m):
            for l in range(self.k):
                self.W[i][l] = self.W[i][l] * (up1[i][l] / down1[i][l])
        for l in range(self.k):
            for j in range(self.n):
                self.H[l][j] = self.H[l][j] * (up2[l][j] / down2[l][j])

    def cost(self):
        X = np.dot(self.W, self.H)
        S = (self.X - X) ** 2
        score = np.sum(S)
        return score


x = []
score = []

# train
L = LSA(X, 3)
L.standard()
for i in range(100):
    L.update()

    x.append(i)
    score.append(L.cost())

plt.figure()
plt.imshow(X)
plt.figure()
plt.imshow(np.dot(L.W, L.H))
plt.figure()
plt.scatter(x, score)
plt.show()
