# 导入numpy库，并简写为np
import numpy as np


# 定义sigmoid函数，用于将任何实数映射到0到1之间的数，常用于二分类问题的输出层
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 数据预处理函数，用于对输入的特征X和标签y进行预处理
def preprocess_data(X, y):
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
    return X, y


# 初始化参数函数，用于初始化权重w和偏置b
def initialize_parameters(n_features):
    # 初始化权重w为一个n_features x 1的零矩阵
    w = np.zeros((n_features, 1))
    # 初始化偏置b为0
    b = 0
    return w, b


# 计算代价函数和梯度的函数，用于计算当前参数下的代价函数值以及梯度值
def compute_cost_gradient(X, y, w, b):
    m = X.shape[0]  # 获取样本数量
    # 计算预测值z和a（通过sigmoid函数）
    z = np.dot(X, w) + b
    a = sigmoid(z)
    # 计算代价函数（交叉熵损失函数）
    cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m
    # 计算梯度：对w和b求偏导
    dw = np.dot(X.T, (a - y)) / m
    db = np.sum(a - y) / m
    return cost, dw, db


# 更新参数函数，用于根据梯度下降算法更新参数w和b
def update_parameters(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw  # 更新权重w
    b = b - learning_rate * db  # 更新偏置b
    return w, b


# 迭代优化函数，用于通过梯度下降算法迭代优化参数w和b，降低代价函数值
def optimize(X, y, w, b, num_iterations, learning_rate):
    costs = []  # 用于存储迭代过程中的代价函数值
    for i in range(num_iterations):  # 迭代num_iterations次
        # 计算当前参数下的代价函数值和梯度值
        cost, dw, db = compute_cost_gradient(X, y, w, b)
        # 根据梯度下降算法更新参数w和b
        w, b = update_parameters(w, b, dw, db, learning_rate)
        if i % 100 == 0:  # 每迭代100次，存储一次代价函数值并打印出来
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))
    return w, b, costs  # 返回优化后的参数w和b，以及迭代过程中的代价函数值


# 预测函数，用于根据优化后的参数w和b对新的输入进行预测（此处的预测不经过数据预处理，因为预测阶段不再需要标准化等操作）
def predict(X, w, b):
    y_pred = (np.dot(X, w) + b > 0.5).astype(int)  # 将预测概率大于0.5的样本预测为正类（1），否则预测为负类（0）
    return y_pred  # 返回预测结果y_pred，是一个与X行数相同的向量，每个元素为0或1。至此，模型构建完成。下面我们将这些函数组合成一个类，方便调用。这个类就是逻辑回归模型。类的初始化函数中，我们可以设定迭代次数和学习率。在fit函数中，我们调用之前定义的函数来训练模型。在predict函数中，我们使用训练好的模型来进行预测。这段代码是完整的逻辑回归模型。在实践中，还需要添加异常处理、输入检查等额外代码。同时，对于大规模数据集，通常还需要使用更复杂的


# 定义一个名为LogisticRegression的类，表示逻辑回归模型
class LogisticRegression:
    # 初始化函数，当创建模型对象时会被调用
    def __init__(self, num_iterations=1000, learning_rate=0.01):
        # 设定迭代次数，默认为1000次
        self.num_iterations = num_iterations
        # 设定学习率，默认为0.01
        self.learning_rate = learning_rate
        # 初始化权重w为None
        self.w = None
        # 初始化偏置b为None
        self.b = None

        # 拟合函数，用于训练模型，根据输入的特征X和标签y来学习模型的参数

    def fit(self, X, y):
        # 获取特征的数量
        n_features = X.shape[1]
        # 调用initialize_parameters函数初始化模型的权重w和偏置b，返回初始化后的权重w和偏置b
        self.w, self.b = initialize_parameters(n_features)
        # 调用optimize函数，通过梯度下降算法优化模型的参数，返回优化后的权重w、偏置b以及训练过程中的代价函数值列表costs
        self.w, self.b, costs = optimize(X, y, self.w, self.b, self.num_iterations, self.learning_rate)
        # 返回代价函数值列表costs
        return costs

    # 预测函数，用于根据学习到的模型参数对新的输入进行预测，返回预测结果
    def predict(self, X):
        # 调用preprocess_data函数对输入的特征X进行预处理，返回处理后的特征X和填充的0的标签y（因为在二分类问题中，y的取值为0或1）
        X, _ = preprocess_data(X, np.zeros(X.shape[0]))
        # 调用predict函数，根据学习到的权重w和偏置b对特征X进行预测，返回预测结果y_pred（一个向量，每个元素为0或1）
        y_pred = predict(X, self.w, self.b)
        # 返回预测结果y_pred
        return y_pred