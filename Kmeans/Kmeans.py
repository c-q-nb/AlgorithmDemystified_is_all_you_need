import numpy as np


class Kmeans:
    # 初始化方法，传入数据和聚类数量
    def __init__(self, data, num_clusters):
        self.data = data  # 存储传入的数据
        self.num_clusters = num_clusters  # 存储聚类的数量

    # 训练方法，进行多次迭代
    def train(self, iterations):
        # 初始化中心点
        centre_ids = Kmeans.centreids_init(self, self.data, self.num_clusters)
        num_examples = self.data.shape[0]  # 获取数据的数量
        closest_centreids_ids = np.empty((num_examples, 1))  # 用于存储每个数据点距离其最近中心点的id

        # 进行多次迭代
        for _ in range(iterations):
            # 找到每个数据点距离其最近中心点的id
            closest_centreids_ids = Kmeans.find_centre_closest(self, self.data, centre_ids)
            # 更新中心点位置
            centreids = Kmeans.centreids_update(self, self.data, closest_centreids_ids, self.num_clusters)
            # 返回最终的中心点和每个数据点对应的最近中心点的id
        return centreids, closest_centreids_ids

        # 更新中心点位置的方法

    def centreids_update(self, data, closest_centreids_ids, num_clusters):
        num_features = data.shape[1]  # 获取数据的特征数量
        centreids = np.zeros((num_clusters, num_features))  # 初始化中心点数组

        # 计算每个中心点的位置（这里是使用每个数据点所属的中心点来计算）
        for centre_id in range(num_clusters):
            closest_id = closest_centreids_ids == centre_id  # 找到属于当前中心点的数据点的id
            centreids[centre_id] = np.mean(data[closest_id.flatten(), :], axis=0)  # 计算中心点位置
        return centreids  # 返回更新后的中心点位置

    # 找到每个数据点距离其最近中心点的id的方法
    def find_centre_closest(self, data, centre_ids):
        num_data = data.shape[0]  # 获取数据点的数量
        num_centreids = centre_ids.shape[0]  # 获取中心点的数量
        closest_centre_ids = np.zeros((num_data, 1))  # 初始化存储每个数据点最近中心点的id的数组

        # 对每个数据点，计算其到所有中心点的距离，并找到最近的中心点
        for data_index in range(num_data):
            distance = np.zeros((num_centreids, 1))  # 初始化存储每个中心点到当前数据点的距离的数组
            for centre_index in range(num_centreids):
                distance_diff = data[data_index, :] - centre_ids[centre_index, :]  # 计算当前数据点到当前中心点的距离
                distance[centre_index] = np.sum(distance_diff ** 2)  # 更新距离数组中的值
            closest_centre_ids[data_index] = np.argmin(distance)  # 找到最近的中心点的id并存储到结果数组中
        return closest_centre_ids  # 返回每个数据点对应的最近中心点的id

    # 初始化中心点的方法，随机选择一些数据点作为初始的中心点
    def centreids_init(self, data, num_clusters):
        num_examples = data.shape[0]  # 获取数据的数量
        ids = np.random.permutation(num_examples)  # 随机打乱数据点的顺序，并获取其id数组
        centreids = data[ids[:num_clusters], :]  # 选择前num_clusters个数据点作为初始的中心点
        return centreids  # 返回初始的中心点位置