import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from DecisionTreeRegression import DecisionTreeRegression, preprocess_data, predict
import matplotlib.pyplot as plt

# def mse(y):
#     return np.mean((y - np.mean(y)) ** 2)
#
# def split_data(X, y, feature_index, threshold):
#     left_mask = X[:, feature_index] <= threshold
#     right_mask = X[:, feature_index] > threshold
#     X_left, y_left = X[left_mask], y[left_mask]
#     X_right, y_right = X[right_mask], y[right_mask]
#     return X_left, y_left, X_right, y_right
#
# def find_best_split(X, y, max_features):
#     best_feature_index, best_threshold = None, None
#     best_mse = float('inf')
#
#     n_features = X.shape[1]
#     if max_features is None:
#         max_features = n_features
#     else:
#         max_features = min(max_features, n_features)
#
#     feature_indices = np.random.choice(n_features, size=max_features, replace=False)
#
#     for feature_index in feature_indices:
#         thresholds = np.unique(X[:, feature_index])
#         for threshold in thresholds:
#             X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
#             mse_left = mse(y_left)
#             mse_right = mse(y_right)
#             total_mse = mse_left + mse_right
#             if total_mse < best_mse:
#                 best_mse = total_mse
#                 best_feature_index = feature_index
#                 best_threshold = threshold
#
#     return best_feature_index, best_threshold
#
# def build_tree(X, y, max_depth, min_samples_split, min_samples_leaf, max_features=None, max_leaf_nodes=None):
#     if max_depth == 0 or X.shape[0] < min_samples_split or np.unique(y).shape[0] == 1 or X.shape[0] < min_samples_leaf:
#         return np.mean(y)
#
#     best_feature_index, best_threshold = find_best_split(X, y, max_features)
#     if best_feature_index is None:
#         return np.mean(y)
#
#     X_left, y_left, X_right, y_right = split_data(X, y, best_feature_index, best_threshold)
#
#     tree = {}
#     tree['feature_index'] = best_feature_index
#     tree['threshold'] = best_threshold
#     tree['left'] = build_tree(X_left, y_left, max_depth - 1, min_samples_split, min_samples_leaf,
#                               max_features, max_leaf_nodes)
#     tree['right'] = build_tree(X_right, y_right, max_depth - 1, min_samples_split, min_samples_leaf,
#                                max_features, max_leaf_nodes)
#
#     return tree
#
# # # 划分数据集
# # def train_test_split(X, y, test_size=0.2):
# #     n_samples = X.shape[0]
# #     n_test = int(n_samples * test_size)
# #     test_indices = np.random.choice(n_samples, n_test, replace=False)
# #     train_indices = np.array(list(set(range(n_samples)) ,set(test_indices)))
# #
# #     X_train, y_train = X[train_indices], y[train_indices]
# #     X_test, y_test = X[test_indices], y[test_indices]
# #
# #     return X_train, X_test, y_train, y_test

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