# 导入相关的库

import math,random

# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split

"""LFM算法"""
class LFM:
    '''
        datafile_path ： movielens数据集路径
        data 数据集
        F ： 隐因子的个数
        lmbd : 正则化系数（为了防止overfitting,添加正则项控制过拟合）
        max_iters:最大的迭代次数，全部的数据集训练多少次
    '''
    def __init__(self,datafile_path,F,alpha=0.1,lmbd=0.1,max_iters = 200):
        '''
        LFM : R(m,n) = P(m,F) * Q(F,n)
        F : 隐因子的个数  P 的每一行代表一个用户对各个隐因子的喜好程度
        Q 每一列代表一个物品在各个隐因子上的概率分布
        '''
        self.datafile_path = datafile_path
        # 获取数据集
        self.train,self.test = self.get_train_test_data()

        self.items_id, self.user_items = self.get_items_user_items()

        self.P = dict()
        self.Q = dict()
        self.datas = self.train[:]

        self.F = F
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iters = max_iters
        # 初始化矩阵P 和 Q
        self.init_P_Q()

        self.get_items_user_items()

    '''随机初始化矩阵P 和 Q'''
    def init_P_Q(self):
        for user,items_rate in self.datas:
            self.P[user] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
            for item,_ in items_rate:
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F) for x in range(self.F)]

    '''使用随机梯度下降法来训练 P 和 Q'''
    def train(self):
        for step in range(self.max_iters):
            for user,items_rates in self.datas:
                for item,rate in items_rates:
                    prv_rate = self.predict(user,item)
                    err_ui = rate - prv_rate
                    # 根据得出的loss ，来更新各个隐因子
                    for f in range(self.F):
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])

            self.alpha *= 0.9

    def predict(self,user,item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F))

    # 获取训练集和测试集
    def get_train_test_data(self):
        # movielens 数据集的路径
        # 记录数据集
        movielensSet = list()

        with open(self.datafile_path,"r") as f:
            for a_user_item in f.readlines():
                user_id,item_id,score,timestamp = a_user_item.split("::")
                # 将用户的评分 和 时间转成整数 str-> int
                score = float(score)
                # 将拆分的数据集放到movielensSet中
                movielensSet.append((user_id,item_id,score))

        # 拆分数据集
        train,test = train_test_split(movielensSet,test_size=0.0,random_state=40)
        # 将train 和 test 转化为字典格式方便使用 list -> dict
        train = self.transform(train)
        test = self.transform(test)

        return train,test

    def transform(self, dataSet):
        data = dict()
        for user_id, item_id, rate in dataSet:
            data.setdefault(user_id, list())
            data[user_id].append((item_id, rate))

        return [(u, i) for u, i in data.items()]

    # 获取物品的所有id 和 用户跟那些物品发生过关系（train）
    def get_items_user_items(self):
        # 记录用户和物品发生过关系 train
        user_items = dict()
        # 记录全部物品的id
        items_id = list()
        for user_id, items in self.train:
            user_items.setdefault(user_id, set())
            for item, _ in items:
                user_items[user_id].add(item)
                items_id.append(item)

        items_id = list(set(items_id))

        return items_id,user_items

    # acc 计算正确率
    def acc(self):
        # 预测正确的个数
        right = 0
        # 测试集数目的总数
        data_sum = 0
        for user,items in self.test:
            # 为用户user 推荐的物品
            user_recommend_items = self.recommend(user)

        pass

    # 为一个用户推荐物品
    '''
        user : 用户 
        k 为用户推荐几个物品
    '''
    def recommend(self,user,k=5):
        rank = dict()
        for item in self.items_id:
            # 该物品用户没有接触过
            if item not in self.user_items[user].keys():
                score = self.predict(user,item)
                rank[item]=score
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:k]).keys()

if __name__ == '__main__':
    #file_path = "/home/lijianmin/github/studyRepository/zhi-neng-tui-jian/datas数据集/MovieLens数据集/ml-1m/ratings.dat"
    file_path = "testData00.txt"
    lfm = LFM(file_path,2)

    #lfm.get_train_test_data()
    for item in ['a','b','c','d']:
        print(item,lfm.predict('A',item))