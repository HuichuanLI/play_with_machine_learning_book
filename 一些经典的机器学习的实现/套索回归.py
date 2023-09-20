# 套索回归
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
'''
套索回归使用l1正则，而岭回归使用l2回归
L1回归会把一些值趋于0，只使用一部分值而不是全部
alpha的值越小使用的值越多，越接近于过拟合
'''
X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y)
# lamda的值是1
la1 = Lasso().fit(X_train, y_train)
print(la1.score(X_train, y_train))
print(la1.score(X_test, y_test))
print("使用的特征值数量：", np.sum(la1.coef_ != 0))
# lamda的值是0.1
la01 = Lasso(alpha=0.1).fit(X_train, y_train)
print(la01.score(X_train, y_train))
print(la01.score(X_test, y_test))
print("使用的特征值数量：", np.sum(la01.coef_ != 0))
# lamda的值是0.001
la001 = Lasso(0.001).fit(X_train, y_train)
print(la001.score(X_train, y_train))
print(la001.score(X_test, y_test))
print("使用的特征值数量：", np.sum(la001.coef_ != 0))
plt.plot(la1.coef_, 's', label='la')
plt.plot(la01.coef_, '*', label='la01')
plt.plot(la001.coef_, '^', label='la001')
plt.hlines(0, 0, len(la1.coef_))
plt.legend(ncol=2, loc=(0, 1.05))
plt.show()