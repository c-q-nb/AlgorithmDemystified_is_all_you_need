from sklearn.datasets import load_iris
from Kmeans import Kmeans
import matplotlib.pyplot as plt
import numpy as np

# 加载Iris数据集
iris = load_iris()
# 从Iris数据集中提取特征，这里只选择了前两列特征
X = iris.data[:, [0, 1]]
# 将目标变量转化为单列矩阵，以便于后续处理
y = iris.target.reshape(-1, 1)
# 将特征矩阵X和目标变量矩阵y合并
result = np.concatenate((X, y), axis=1)

# 设置聚类的数量为3
num_clusters = 3
# 设置最大迭代次数为50次，用于K-means算法
max_iteritions = 50

# 初始化K-means类，输入参数为特征矩阵X和聚类数量num_clusters
k_means = Kmeans(X, num_clusters)

# 使用K-means类中的train方法进行训练，输入参数为最大迭代次数max_iteritions，输出中心点和最近中心点的id
centre_ids, closest_centroids = k_means.train(max_iteritions)

# 创建一个新的图形，大小为12x5，分为1行2列的子图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

# 在第一个子图中，分别绘制三种iris类型的数据点，分别用不同的颜色表示
for iris_type in (0, 1, 2):
    plt.scatter(result[:, 0][result[:, -1] == iris_type], result[:, 1][result[:, -1] == iris_type])

# 设置第一个子图的标题为'Label known'
plt.title('Label known')
# 设置第一个子图的图例
plt.legend()

# 设置第二个子图，显示通过K-means算法得到的聚类结果
plt.subplot(1, 2, 2)

# 遍历每个聚类的中心点
for centre_id, centreid in enumerate(centre_ids):
    # 获取当前聚类中的所有样本的索引
    current_examples_index = (closest_centroids == centre_id).flatten()

    # 使用不同的颜色绘制当前聚类中的所有样本
    plt.scatter(result[:, 0][current_examples_index], result[:, 1][current_examples_index])

    # 使用黑色x标记当前聚类的中心点
    plt.scatter(centreid[0], centreid[1], c='black', marker='x')

# 设置第二个子图的标题为'label kmeans'
plt.title('label kmeans')
# 设置第二个子图的图例，显示每个聚类的颜色和中心点的标记方式
plt.legend()

# 显示整个图形
plt.show()
