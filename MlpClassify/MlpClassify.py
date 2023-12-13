import numpy as np


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


class MlpNetwork:  # 定义一个名为NeuralNetwork的类
    def __init__(self, input_size, hidden_size, output_size):  # 初始化函数，接受输入、隐藏层和输出的大小
        self.input_size = input_size  # 存储输入大小
        self.hidden_size = hidden_size  # 存储隐藏层大小
        self.output_size = output_size  # 存储输出大小

        self.weights1 = np.random.randn(self.input_size, self.hidden_size)  # 初始化权重1，用于输入到隐藏层的连接
        self.bias1 = np.zeros((1, self.hidden_size))  # 初始化偏置1，用于隐藏层
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)  # 初始化权重2，用于隐藏层到输出的连接
        self.bias2 = np.zeros((1, self.output_size))  # 初始化偏置2，用于输出层

    def forward(self, X):  # 前向传播函数，接受输入数据X
        self.z1 = np.dot(X, self.weights1) + self.bias1  # 计算第一层的加权输入和偏置
        self.a1 = self.sigmoid(self.z1)  # 使用sigmoid函数激活第一层
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2  # 计算第二层的加权输入和偏置
        self.a2 = self.sigmoid(self.z2)  # 使用sigmoid函数激活第二层
        return self.a2  # 返回第二层的输出

    def sigmoid(self, x):  # sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def loss(self, X, y):  # 损失函数，计算模型预测值与真实值之间的差异
        m = X.shape[0]  # 获取样本数量
        predicted_output = self.forward(X)  # 获取模型预测值
        cost = -1 / m * np.sum(y * np.log(predicted_output) + (1 - y) * np.log(1 - predicted_output))  # 使用交叉熵损失计算成本
        return cost  # 返回成本

    def backward(self, X, y):  # 定义反向传播函数
        m = X.shape[0]  # 获取输入数据X的样本数量

        # 计算第二层（输出层）的误差
        dZ2 = self.a2 - y
        # 计算第二层权重的梯度。这里使用了矩阵乘法来计算每个样本的误差对权重的梯度
        dW2 = 1 / m * np.dot(self.a1.T, dZ2)
        # 计算第二层偏置的梯度。np.sum(dZ2, axis=0, keepdims=True)确保了每个样本的误差对偏置的梯度是独立的
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
        # 计算第一层（隐层）的误差。这里首先计算了误差在第二层与第一层之间的传递，然后乘以sigmoid函数的导数，用于下一步的权重梯度计算
        dZ1 = np.dot(dZ2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        # 计算第一层权重的梯度。这里使用了矩阵乘法来计算每个样本的误差对权重的梯度
        dW1 = 1 / m * np.dot(X.T, dZ1)
        # 计算第一层偏置的梯度。np.sum(dZ1, axis=0, keepdims=True)确保了每个样本的误差对偏置的梯度是独立的
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
        # 返回各参数的梯度，供后续的权重和偏置更新使用
        return dW1, db1, dW2, db2

    def sigmoid_derivative(self, x):  # sigmoid函数的导数
        return x * (1 - x)

    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):  # 更新参数的函数
        self.weights1 -= learning_rate * dW1  # 更新第一层的权重和偏置
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dW2  # 更新第二层的权重和偏置
        self.bias2 -= learning_rate * db2

    def train(self, X, y, num_epochs, learning_rate):  # 定义训练函数
        for epoch in range(num_epochs):  # 开始循环，进行num_epochs次迭代
            output = self.forward(X)  # 对输入数据X进行前向传播，得到输出结果

            dW1, db1, dW2, db2 = self.backward(X, y)  # 计算反向传播的梯度，得到各参数的梯度
            self.update_parameters(dW1, db1, dW2, db2, learning_rate)  # 使用梯度下降法更新权重和偏置
            cost = self.loss(X, y)  # 计算当前迭代的损失
            if epoch % 100 == 0:  # 如果当前是100的倍数（每100轮）
                print(f"Epoch {epoch}, loss: {cost}")  # 打印当前迭代的轮数和损失