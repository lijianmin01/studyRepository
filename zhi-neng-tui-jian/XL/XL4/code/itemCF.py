# 导入需要的库
import os,math,random,json
# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split

# 2、使用movielens数据集实现基于时间衰减的ItemCF算法，并附代码！
class ItemCF:
    # datafile_path：传入的movielens数据集的路径
    def __init__(self, datafile_path):
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
        self.train, self.test, self.max_data = self.load_movielens()
        # 用户间的相似度
        self.items_sim = self.ItemSimilarity()

    #  加载movielens数据集
    def load_movielens(self):
        print("加载数据集 并 拆分数据集")
        # 记录数据集
        movielensSet = list()
        # 记录用户与商品发生关系的最晚时间
        max_timestamp = 0

        # 打开数据集
        with open(self.datafile_path, "r") as f:
            for a_user_item in f.readlines():
                user_id, item_id, score, timestamp = a_user_item.split("::")
                # 将用户的评分 和 时间转成整数 str-> int
                score, timestamp = int(score), int(timestamp)
                # 将拆分的数据集放到movielensSet中
                movielensSet.append((user_id, item_id, score, timestamp))
                # 寻找最大时间（记录用户与商品发生关系的最晚时间）
                if timestamp > max_timestamp:
                    max_timestamp = timestamp

        # 拆分数据集
        # 训练集：测试集  = 9:1
        train, test = train_test_split(movielensSet, test_size=0.1, random_state=40)

        # 将train 和 test 转化为字典格式方便使用 list -> dict
        train_dict = self.transform(train)
        test_dict = self.transform(test)

        return train_dict, test_dict, max_timestamp

    # list - > dict
    def transform(self, data_list):
        data_dict = dict()
        for user_id, item_id, record, timestamp in data_list:
            data_dict.setdefault(user_id, {}).setdefault(item_id, {})
            # 写入时间和评分
            data_dict[user_id][item_id]['rate'] = record
            data_dict[user_id][item_id]['time'] = timestamp

        return data_dict

    # 计算物品之间的相似度
    def ItemSimilarity(self):
        print("开始计算物品之间的相似度...")
        if os.path.exists("../json/item_sim.json"):
            print("从文件中加载数据")
            item_sim = json.load(open("../json/item_sim.json", "r"))
            return item_sim
        else:
            print("计算用户之间的相似度")
            # 物品与物品之间的相似矩阵
            item_sim = dict()
            # 记录每个物品有多少用户产生过行为
            N = dict()
            # cuv
            count = dict()
            for user, items in self.train.items():
                for item_id in items.keys():
                    N.setdefault(item_id, 0)
                    if self.train[str(user)][item_id]["rate"] > 0.0:
                        N[item_id] += 1
                    for item_id2 in items.keys():
                        count.setdefault(item_id, {}).setdefault(item_id2, 0)
                        if item_id != item_id2 and (self.train[user][item_id]["rate"] > 0.0) and (
                                self.train[user][item_id2]["rate"] > 0.0):
                            count[item_id][item_id2] += 1.0 / (1.0 + self.alpha * abs(
                                self.train[user][item_id]["time"] - self.train[user][item_id2]["time"]))

            # count - > item_sim
            for i, related_items in count.items():
                item_sim.setdefault(i, {})
                for j, cuv in related_items.items():
                    item_sim[i].setdefault(j, 0)
                    item_sim[i][j] = cuv / math.sqrt(N[i] * N[j])

            # 将计算出的结构，写入文件中
            json.dump(item_sim, open("../json/item_sim.json", "w"))
            print("物品之间的相似度已经写入文件")

            return item_sim

    # 为用户userA 进行物品推荐
    # # k 选取前K个用户,跟用户UserA 兴趣想近的
    # # n_items 为用户推荐n_items 个物品，即打分最高的前n_items 个物品
    def recommend(self, user, k=8, n_items=40):
        # items_rank 用来记录物品分数排名
        items_rank = dict()
        # 获取用户感兴趣的物品
        interacted_items = self.train.get(user, {})
        for user_item, rate_time in interacted_items.items():
            # 用用户喜欢的商品中，找出与用户喜欢的商品想近的物品itemA
            for itemA, wj in sorted(self.items_sim[user_item].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if user_item == itemA:
                    continue
                items_rank.setdefault(itemA, 0.0)
                items_rank[itemA] += rate_time["rate"] * wj * (
                            1.0 / (1.0 + self.beta * abs(self.max_data - rate_time["time"])))

        # 返回分数最高的钱n_items 个物品
        return dict(sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[0:n_items])

    # 计算预测正确率
    def precision(self, k=8, n_items=10):
        hit = 0.0
        precision_nums = 0
        # 用test中随机挑选10名用户
        for user in random.sample(self.test.keys(), 10):
            tu = self.test.get(user, {})
            rank = self.recommend(user, k=k, n_items=n_items)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1

            precision_nums += n_items

        return hit / (precision_nums * 1.0)


if __name__ == '__main__':
    file_path = "../ml-1m/ratings.dat"
    cf = ItemCF(file_path)
    result = cf.recommend("1")
    print("user '1' recommend result is {}".format(result))

    pre = cf.precision()
    print("pre is {}".format(pre))