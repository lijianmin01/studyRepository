# 导入需要的库
import os,math,random,json
# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split


# 1、使用movielens数据集实现基于时间衰减的UserCF算法，并附代码！
class UserCF:
    # datafile_path：传入的movielens数据集的路径
    def __init__(self,datafile_path):
        # ɑ是时间衰减参数，它的取值在不同系统中不同。
        # 如果 一个系统用户兴趣变化很快，就应该取比较大的ɑ ，反之需要取比较小的ɑ 。
        # 计算SIM 时用到
        self.alpha = 0.5
        # 时间信息对预测公式的影响
        # 用户现在的行为应该和用户最近的行为关系更大
        # 计算 p(u,i) 时
        self.beta = 0.8
        self.datafile_path = datafile_path
        # 加载数据集，拆分数据集
        self.train,self.test,self.max_data = self.load_movielens()
        # 用户间的相似度
        self.users_sim = self.UserSimilarity()

    #  加载movielens数据集
    def load_movielens(self):
        print("加载数据集 并 拆分数据集")
        # 记录数据集
        movielensSet = list()
        # 记录用户与商品发生关系的最晚时间
        max_timestamp = 0

        # 打开数据集
        with open(self.datafile_path,"r") as f:
            for a_user_item in f.readlines():
                user_id,item_id,score,timestamp = a_user_item.split("::")
                # 将用户的评分 和 时间转成整数 str-> int
                score, timestamp = int(score),int(timestamp)
                # 将拆分的数据集放到movielensSet中
                movielensSet.append((user_id,item_id,score,timestamp))
                # 寻找最大时间（记录用户与商品发生关系的最晚时间）
                if timestamp > max_timestamp:
                    max_timestamp = timestamp

        # 拆分数据集
        # 训练集：测试集  = 9:1
        train,test = train_test_split(movielensSet,test_size=0.1,random_state=40)

        # 将train 和 test 转化为字典格式方便使用 list -> dict
        train_dict = self.transform(train)
        test_dict = self.transform(test)

        return train_dict, test_dict, max_timestamp

    # list - > dict
    def transform(self,data_list):
        data_dict = dict()
        for user_id, item_id, record, timestamp in data_list:
            data_dict.setdefault(user_id,{}).setdefault(item_id,{})
            # 写入时间和评分
            data_dict[user_id][item_id]['rate'] = record
            data_dict[user_id][item_id]['time'] = timestamp

        return data_dict

    # 计算用户与用户之间的相似度，采用惩罚热门商品和优化算法的复杂度（建立倒排表）
    def UserSimilarity(self):
        print("计算用户与用户之间的相似度...")
        # 这里由于计算量过大，我们二种方式：
        # # 第一次运算  ，之间计算，并保存计算所得的数据
        # # 直接加载之前计算所得的数据
        """
        使用json函数需要导入json库：import json
        json.dumps()    将python对象编码成JSON字符串（可以这么理解，json.dumps()函数是将字典转化为字符串）
        json.loads()    将已编码的JSON字符串解码为python对象（可以这么理解，json.loads()函数是将字符串转化为字典）
        """
        if os.path.exists("../../../../../github_not_data/json/user_sim.json"):
            print("从文件中加载用户之间的相似度")
            user_sim = json.load(open("../../../../../github_not_data/json/user_sim.json"))
            return user_sim
        else:
            print("计算用户之间的相似度")
            # 记录item与那些user 发生过关系
            item_eval_by_users = dict()
            # 处理train数据
            for user_id,items in self.train.items():
                for item_id in items.keys():
                    item_eval_by_users.setdefault(item_id,set())
                    # 用户评分 > 0
                    if self.train[user_id][item_id]["rate"] > 0.0:
                        item_eval_by_users[item_id].add(user_id)

            # 构建倒排表
            C = dict()
            # N 记录 用户跟浏览过几种产品
            N = dict()

            for item_id,users in item_eval_by_users.items():
                for user_id in users:
                    N.setdefault(user_id,0)
                    N[user_id] += 1
                    C.setdefault(user_id,{})
                    for v in users:
                        C[user_id].setdefault(v,0.0)
                        # 如果是同一个人
                        if user_id == v:
                            continue
                        # 热门物品的惩罚  * 时间的惩罚  a
                        C[user_id][v] += (1.0 / math.log(1+len(users))) *( 1.0 / (1.0 + self.alpha * abs(self.train[user_id][item_id]["time"] - self.train[v][item_id]["time"])))

            # 用户的相似矩阵  Wij
            user_sim = dict()
            for u,related_users in C.items():
                user_sim.setdefault(u,{})
                for v,cuv in related_users.items():
                    if u == v:
                        continue
                    user_sim[u].setdefault(v,0.0)
                    # wij = cuv/math.sqrt(N[u]*N[v]
                    user_sim[u][v] = cuv/math.sqrt(N[u]*N[v])

            # 将计算出的结构，写入文件中
            json.dump(user_sim, open("../../../../../github_not_data/json/user_sim.json", "w"))
            print("用户之间的相似度已经写入文件")
            # 返回用户之间的相似度
            return user_sim

    # 为用户userA 进行物品推荐
    # # k 选取前K个用户,跟用户UserA 兴趣想近的
    # # n_items 为用户推荐n_items 个物品，即打分最高的前n_items 个物品
    def recommend(self,userA,k=8,n_items=40):
        # items_rank 用来记录物品分数排名
        items_rank = dict()
        # 获取用户感兴趣的物品
        interacted_items = self.train.get(userA, {})
        # 对用户之间的相似度进行排序
        # print("recommend")
        # print(sorted(self.users_sim[userA].items,key=lambda d:d[1],reverse=True))
        # exit(0)
        for userB,wab in sorted(self.users_sim[userA].items(),key=lambda d:d[1],reverse=True)[0:k]:
            for item,rv in self.train[userB].items():
                for i in interacted_items:
                    continue
                items_rank.setdefault(item,0.0)

                items_rank[item] += wab * rv["rate"] * (1.0/(1.0+self.beta * (self.max_data - abs(rv["time"]))))

        return dict(sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[0:n_items])

    # 计算预测正确率
    def precision(self,k=8,n_items=10):
        hit = 0.0
        precision_nums = 0
        # 用test中随机挑选10名用户
        for user in random.sample(self.test.keys(),10):
            tu = self.test.get(user,{})
            rank = self.recommend(user, k=k, n_items=n_items)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1

            precision_nums += n_items

        return hit / (precision_nums * 1.0)

if __name__ == '__main__':
    # file_path = "../ml-1m/ratings.dat"
    file_path = "../../../../../github_not_data/ml-1m/ratings.dat"
    cf = UserCF(file_path)
    result = cf.recommend("1")
    print("user '1' recommend result is {}".format(result))

    pre = cf.precision()
    print("pre is {}".format(pre))