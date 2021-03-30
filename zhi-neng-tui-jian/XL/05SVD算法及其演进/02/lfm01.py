import numpy as np
from sklearn.model_selection import train_test_split


class LFM:
    """
    file_path:文件路径
    k : 隐因子个数
    lr : 学习率
    re : 正则化系数
    max_iter: 数据迭代次数
    """

    def __init__(self, file_path, k, lr=0.1, re=0.1, max_iter=260):
        self.file_path = file_path
        self.k = k
        self.lr = lr
        self.re = re
        self.max_iter = max_iter
        # 获取训练数据和测速数据,所有的物品id
        self.rating_data, self.test, self.all_items_id = self.get_data()
        # 随机初始化矩阵P 和 Q
        self.P, self.Q = self.init_P_Q()

    """获取数据集并拆分数据集  train:test = 19:1"""

    def get_data(self):
        movieSet = list()
        all_items = list()
        with open(self.file_path, "r") as f:
            # 使用前 20000 条数据  ，要不电脑跑不动
            for line in f.readlines()[:20000]:
                user_id, item_id, score, _ = line.split("::")
                movieSet.append((user_id, item_id, float(score)))
                all_items.append(item_id)

        train, test = train_test_split(movieSet, test_size=0.05, random_state=13)

        train_dict = self.transform(train)

        return train_dict, test, list(set(all_items))

    """list -> dict"""

    def transform(self, data_list):
        data_dict = dict()
        for user_id, item_id, record in data_list:
            data_dict.setdefault(user_id, dict())
            data_dict[user_id][item_id] = record

        return data_dict

    """随机初始化矩阵P 和 Q
        P : 用户因子矩阵
        Q ：物品因子矩阵
    """

    def init_P_Q(self):
        P = dict()
        Q = dict()
        for user in self.rating_data.keys():
            P[user] = np.random.random(size=(1, self.k))

        for item in self.all_items_id:
            Q[item] = np.random.random(size=(1, self.k))

        return P, Q

    """使用随机梯度下降法，更新矩阵 P 和 Q """

    def train(self):
        print("开始训练")
        for iter in range(self.max_iter):
            loss = 0.0
            for user, items in self.rating_data.items():
                for item, score in items.items():
                    prv_score = np.dot(self.P[user], self.Q[item].T)[0][0]
                    err_score = score - prv_score
                    loss += err_score * err_score
                    self.P[user] += self.lr * (err_score * self.Q[item] - self.re * self.P[user])
                    self.Q[item] += self.lr * (err_score * self.P[user] - self.re * self.Q[item])

            if (iter+1)%20==0:
                print("第{}轮训练，损失为{}".format(iter+1,loss))
            self.lr *= 0.9
        print("训练结束")

    """预测用户user 对item 的评分"""

    def predict(self, user, item):
        return np.dot(self.P[user], self.Q[item].T)[0]

    """为用户推荐物品
        user : 被推荐的用户
        k : 推荐的物品数目
    """

    def recommend(self, user, k=5):
        rank = dict()
        for item in self.all_items_id:
            if item not in self.rating_data[user].keys():
                rank[item] = np.dot(self.P[user], self.Q[item].T)[0]

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:k])

    """使用测试集 来评估模型
        评估指标： mean(sum(abs（用户真实打分 - 模型预测分数)))
    """

    def evaluate_model(self):
        score = 0.0
        data_sum = len(self.test)
        for user, item, rate in self.test:
            if item not in self.all_items_id:
                continue
            prv = self.predict(user, item)
            score += abs(rate - prv)

        score /= data_sum
        print("评价指标分数为：{}".format(score))
        return score


if __name__ == '__main__':
    file_path = '../ratings.dat'
    lfm = LFM(file_path, 5)
    lfm.train()
    print("为用户 1 推荐电影(推荐5部电影)")
    items = lfm.recommend('1', k=5)
    for m, rate in items.items():
        print("电影：{}，分数：{}".format(m, rate))

    lfm.evaluate_model()



