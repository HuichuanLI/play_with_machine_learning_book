import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
class BP(object):
    def __init__(self, layers, activation='sigmoid', learning_rate=0.01):
self.layers = layers
self.learning_rate = learning_rate
self.caches = {}
self.grades = {}
        if activation == 'sigmoid':
self.activation = sigmoid
self.dactivation = dsigmoid
self.parameters = {}
        for i in range(1, len(self.layers)):
self.parameters["w"+str(i)] = np.random.random((self.layers[i], self.layers[i-1]))
self.parameters["b"+str(i)] = np.zeros((layers[i],1))
    def forward(self, X):
        a = []
        z = []
a.append(X)
z.append(X)
len_layers = len(self.parameters) // 2
            for i in range(1, len_layers):
                z.append(self.parameters["w"+str(i)] @ a[i-1] + self.parameters["b"+str(i)])
a.append(sigmoid(z[-1]))
z.append(self.parameters["w"+str(len_layers)] @ a[-1] + self.parameters["b"+str(len_layers)])
a.append(z[-1])
self.caches['z'] = z
self.caches['a'] = a
          return self.caches, a[-1]
    def backward(self, y):
        a = self.caches['a']
        m = y.shape[1]
len_layers = len(self.parameters) // 2
self.grades["dz"+str(len_layers)] = a[-1]-y
self.grades["dw"+str(len_layers)] = self.grades["dz"+str(len_layers)].dot(a[-2].T) / m
self.grades["db"+str(len_layers)] = np.sum(self.grades["dz"+str(len_layers)], axis=1, keepdims=True) / m
        for i in reversed(range(1, len_layers)):
self.grades["dz"+str(i)] = self.parameters["w"+str(i+1)].T.dot(self.grades["dz"+str(i+1)]) * dsigmoid(self.caches["z"][i])
self.grades["dw"+str(i)] = self.grades["dz"+str(i)].dot(self.caches["a"][i-1].T)/m
self.grades["db"+str(i)] = np.sum(self.grades["dz"+str(i)],axis = 1,keepdims = True) /m
        #update weights and bias
        for i in range(1, len(self.layers)):
self.parameters["w"+str(i)] -= self.learning_rate * self.grades["dw"+str(i)]
self.parameters["b"+str(i)] -= self.learning_rate * self.grades["db"+str(i)]
    def compute_loss(self, y):
        return np.mean(np.square(self.caches['a'][-1]-y))
def test():
    x = np.arange(0.0,1.0,0.01)
    y =20* np.sin(2*np.pi*x)
plt.scatter(x,y)
    x = x.reshape(1, 100)
    y = y.reshape(1, 100)
    bp = BP([1, 6, 1], learning_rate = 0.01)
    for i in range(1, 50000):
        caches, al = bp.forward(x)
bp.backward(y)
        if(i%50 == 0):
            print(bp.compute_loss(y))
            plt.scatter(x, al)
plt.show()
test()