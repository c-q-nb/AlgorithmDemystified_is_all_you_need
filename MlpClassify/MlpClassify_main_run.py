from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from MlpClassify import MlpNetwork, preprocess_data
import numpy as np

# 加载数据集 取两种类别的燕尾花
iris = load_iris()
X = iris.data[0:100, [0, 1]]
y = iris.target[0:100].reshape(-1, 1)
# 数据异常处理，数据特征工程，特征筛选等
# 具体的根据业务需求和数据进行相关的异常值，特征筛选，特征工程等
X = preprocess_data(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型并训练
model = MlpNetwork(input_size=2, hidden_size=32, output_size=1)
model.train(X_train, y_train, num_epochs=1000, learning_rate=0.1)

# 进行预测
y_pred = model.forward(X_test)
y_pred = (y_pred>0.5).astype(int)
# 计算准确率、召回率和F1值
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
