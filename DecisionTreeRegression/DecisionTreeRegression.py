import numpy as np


# 数据预处理函数，用于对输入的特征X和标签y进行预处理
def preprocess_data(X):
    # 处理缺失值
    # 如果是数值类型的缺失值，可以用平均值、中位数或众数填充
    # 这里我们用平均值填充
    # X = np.where(np.isnan(X), X.mean(), X)

    # 处理异常值
    # 可以使用箱线图法或3σ原则进行异常值的识别和删除
    # 这里我们使用箱线图法进行异常值的识别和删除
    # X = np.clip(X, np.percentile(X, 25), np.percentile(X, 75))
    # X = np.where(np.abs(X - np.mean(X)) > 3 * np.std(X), np.mean(X), X)

    # 根据需求选择合适的特征
    # 可以使用相关性分析、卡方检验、互信息等方法进行特征选择
    # 这里我们使用相关性分析进行特征选择，保留相关性大于0.5的特征
    # corr_matrix = np.corrcoef(X, rowvar=False)
    # features = np.where(corr_matrix > 0.5, True, False)
    # X = X[:, features]

    # 数据标准化：将X的每一列（特征）进行标准化，使其均值为0，标准差为1
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X
def mse(y):
    return np.mean((y - np.mean(y)) ** 2)

# 定义一个函数，用于根据指定的特征和阈值将数据集分割为两部分
def split_data(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold  # 创建左侧数据的掩码
    right_mask = X[:, feature_index] > threshold  # 创建右侧数据的掩码
    X_left, y_left = X[left_mask], y[left_mask]  # 提取左侧数据
    X_right, y_right = X[right_mask], y[right_mask]  # 提取右侧数据
    return X_left, y_left, X_right, y_right  # 返回分割后的数据

    # 定义一个函数，用于找到最佳的特征和阈值进行数据分割
def find_best_split(X, y, max_features):
    best_feature_index, best_threshold = None, None  # 初始化最佳特征和阈值
    best_mse = float('inf')  # 初始化最小的均方误差为无穷大

    n_features = X.shape[1]  # 获取特征的数量
    if max_features is None:  # 如果max_features为None
        max_features = n_features  # 则将max_features设为全部特征
    else:
        max_features = min(max_features, n_features)  # 否则，将max_features设为其与特征数量的较小值

    feature_indices = np.random.choice(n_features, size=max_features, replace=False)  # 从所有特征中随机选择部分特征

    for feature_index in feature_indices:  # 对每个选择的特征进行遍历
        thresholds = np.unique(X[:, feature_index])  # 获取该特征的所有唯一值作为可能的阈值
        for threshold in thresholds:  # 对每个可能的阈值进行遍历
            X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)  # 使用当前特征和阈值分割数据
            mse_left = mse(y_left)  # 计算左侧数据的均方误差
            mse_right = mse(y_right)  # 计算右侧数据的均方误差
            total_mse = mse_left + mse_right  # 计算总的均方误差
            if total_mse < best_mse:  # 如果当前的总均方误差小于之前的最佳均方误差
                best_mse = total_mse  # 则更新最佳均方误差
                best_feature_index = feature_index  # 并更新最佳特征索引
                best_threshold = threshold  # 和最佳阈值

    return best_feature_index, best_threshold  # 返回最佳特征和阈值

    # 定义决策树回归模型的核心函数
def DecisionTreeRegression(X, y, max_depth, min_samples_split, min_samples_leaf, max_features=None,
                           max_leaf_nodes=None):
    if max_depth == 0 or X.shape[0] < min_samples_split or np.unique(y).shape[0] == 1 or X.shape[
        0] < min_samples_leaf:  # 如果深度为0或样本数小于min_samples_split或y中只有一个类别或样本数小于min_samples_leaf，则直接返回y的平均值作为预测结果
        return np.mean(y)

    best_feature_index, best_threshold = find_best_split(X, y, max_features)  # 找到最佳的特征和阈值进行数据分割
    if best_feature_index is None:  # 如果找不到最佳特征，则直接返回y的平均值作为预测结果
        return np.mean(y)

    # 根据最佳特征和阈值将数据集分割为两部分
    X_left, y_left, X_right, y_right = split_data(X, y, best_feature_index, best_threshold)

    # 创建一个字典来存储决策树的模型信息
    model = {}

    # 将最佳特征索引、阈值和左右子树的预测结果存储在模型字典中
    model['feature_index'] = best_feature_index
    model['threshold'] = best_threshold
    model['left'] = DecisionTreeRegression(X_left, y_left, max_depth - 1, min_samples_split, min_samples_leaf,
                                           max_features, max_leaf_nodes)
    model['right'] = DecisionTreeRegression(X_right, y_right, max_depth - 1, min_samples_split, min_samples_leaf,
                                            max_features, max_leaf_nodes)

    # 返回构建好的决策树模型
    return model


# 使用决策树模型对新的输入数据进行预测
def predict(model, x):
    # 如果模型是一个浮点数（即叶节点的值），则直接返回该值
    if isinstance(model, float):
        return model
    # 获取当前决策树的特征索引和阈值
    feature_index = model['feature_index']
    threshold = model['threshold']
    # 根据特征的值决定预测结果，是左子树还是右子树
    if x[feature_index] <= threshold:
        return predict(model['left'], x)
    else:
        return predict(model['right'], x)
