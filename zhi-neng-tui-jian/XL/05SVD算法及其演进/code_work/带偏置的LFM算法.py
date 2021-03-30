import math,random
# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split

class LFM:
    # 初始化
    """
        file_path:文件路径
        F ： 隐因子的个数
        alpha: 学习率
        lmbd: 正则化
        max_iter： 最大迭代次数
    """
    def __init__(self,file_path,F,alpha=0.1,lmbd=0.1,max_iter = 500):
        self.file_path = file_path
        self.F = F
        self.P = dict()  # R = P Q T  Q相当于Q的转置
        self.Q = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.bu = dict()
        self.bi = dict()
        self.mu = 0.0
        self.rating_data,self.test = self.get_rating_data()
        # 获取rating_data 中所有的物品id
        self.all_items = self.get_all_items()

        """随机初始化P 和 Q"""
        cnt = 0
        for user, rates in self.rating_data.items():
            self.P[user] = [random.random() / math.sqrt(self.F) for j in range(self.F)]
            self.bu[user] = 0
            cnt += len(rates)
            for item, rate in rates.items():
                self.mu += rate
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F) for x in range(self.F)]

                self.bi[item] = 0

        self.mu = self.mu/cnt

    # 划分数据集 9 ：1
    def get_rating_data(self):
        movieSet = list()
        with open(self.file_path,"r") as f:
            for line in f.readlines():
                user_id, item_id, score, timestamp = line.split("::")
                score = float(score)
                # 将拆分的数据集放到movielensSet中
                movieSet.append((user_id,item_id,score))

        # 拆分数据集
        # 训练集：测试集  = 9:1
        train, test = train_test_split(movieSet, test_size=0.1, random_state=40)

        # 将train 和 test 转化为字典格式方便使用 list -> dict
        train_dict = self.transform(train)

        return train_dict,test

    # 随机梯度下降法训练参数P和Q
    def train(self):
        for step in range(self.max_iter):
            for user,rates in self.rating_data.items():
                for item,rate in rates.items():
                    hat_rate = self.predict(user,item)
                    err_rate = rate-hat_rate
                    # 更新偏置
                    self.bu[user] += self.alpha * (err_rate - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_rate - self.lmbd * self.bi[item])
                    for f in range(self.F):
                        self.P[user][f] += self.alpha * (err_rate * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_rate * self.P[user][f] - self.lmbd * self.Q[item][f])

            self.alpha *= 0.9

    # 预测用户user 对item 的评分
    def predict(self, user, item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F)) + self.bu[user] + self.bi[item] + self.mu

    # 讲 list -> dict
    def transform(self, data):
        data_dict = dict()
        for user_id, item_id, record in data:
            data_dict.setdefault(user_id,dict())
            data_dict[user_id][item_id]=record

        return data_dict

    # 获取rating_data 中所有的物品id
    def get_all_items(self):
        items_id = list()
        for user,items in self.rating_data.items():
            for item,_ in items.items():
                items_id.append(item)

        return list(set(items_id))

    # 为用户推荐物品 为用户user推荐k个物品
    def recommend(self,user,k=5):
        rank = dict()
        for item in self.all_items:
            if item not in self.rating_data[user].keys():
                rank[item] = self.predict(user,item)

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:k])

    # 对模型进行自我评估（test）
    # 这里我们采用 （  sqrt((预测分数-实际分数)^2)    得到全部的总数  /  数据的总数  ）
    def evaluate_model(self):
        score = 0.0
        data_sum = len(self.test)
        for user,item,rate in self.test:
            prv = self.predict(user,item)
            score += math.sqrt(math.pow((prv-rate),2))
        score /= data_sum
        return score

if __name__ == '__main__':
    print("带偏置的LFM算法:")
    file_path = "ratings.dat"
    lfm = LFM(file_path, 2)
    lfm.train()
    print("为用户 1 推荐电影(推荐5部电影)")
    items = lfm.recommend('1')
    for m,rate in items.items():
        print("电影：{}，分数：{}".format(m,rate))
    print("改模型在测试集上的评价分数为{}".format(lfm.evaluate_model()))
