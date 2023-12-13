import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from DecisionTreeRegression import DecisionTreeRegression, preprocess_data, predict
import matplotlib.pyplot as plt

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 数据异常处理，数据特征工程，特征筛选等
# 具体的根据业务需求和数据进行相关的异常值，特征筛选，特征工程等
X = preprocess_data(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 构建决策树模型
max_depth = 20  # 最大深度
min_samples_split = 2  # 内部节点的最小样本数
min_samples_leaf = 2  # 叶子节点的最小样本数
max_features = None  # 考虑的最大特征数（默认全部特征）
max_leaf_nodes = None  # 最大叶子节点数

model = DecisionTreeRegression(X_train, y_train, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes)

y_pred = np.array([predict(model, x) for x in X_test])
print(y_pred)

# 创建一个图形框
fig = plt.figure(figsize=(8, 6))

# 在图形框里只画一幅图
ax = fig.add_subplot(111)

# 在图上画数据点
ax.plot(y_pred, 'bo', label='Predictions')
ax.plot(y_test, 'ro', label='Test Data')

# 设置图形的标题以及x和y轴的标签
ax.set_title('Predictions vs Test Data')
ax.set_xlabel('Index')
ax.set_ylabel('Value')

# 设置图例
ax.legend()

# 显示图形
plt.show()