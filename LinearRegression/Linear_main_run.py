import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from LinearRegression import LinearRegression, preprocess_data
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据集
boston = load_boston()
X = boston.data
y = boston.target.reshape(-1, 1)

# 数据异常处理，数据特征工程，特征筛选等
# 具体的根据业务需求和数据进行相关的异常值，特征筛选，特征工程等
X = preprocess_data(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

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