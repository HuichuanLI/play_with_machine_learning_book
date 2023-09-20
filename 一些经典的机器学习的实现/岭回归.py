# 岭回归
from sklearn.datasets import load_boston  #sklearn波士顿房价预测数据接口
from sklearn.model_selection import train_test_split  #划分数据集
from sklearn.preprocessing import StandardScaler    #数据标准化
from sklearn.linear_model import Ridge  #预估器（正规方程）、预估器（梯度下降学习）、岭回归
from sklearn.metrics import mean_squared_error  #均方误
from sklearn.externals import joblib   #模型的加载与保存
def linear():
    # 1）获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)
    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4）预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)
    # 保存模型
    joblib.dump(estimator, "my_ridge.pkl")
    # 加载模型 使用时注销 4）预估器 和 保存模型
    #     estimator = joblib.load("my_ridge.pkl"
    # 5）得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)
    # 6）模型评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)
    return None
if __name__ == "__main__":
    linear()

