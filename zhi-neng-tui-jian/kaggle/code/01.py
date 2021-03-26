# 基于物品的协同过滤算法


import random
import math


class ItemCF:

    def __init__(self,user_score_dict):
        self.begin_user_score_dict = user_score_dict
        self.all_artists = self.get_all_artists()
        self.user_score_dict = self.deal_user_score_dict()
        self.items_sim = self.itemSimilarityBest()

    # 处理初始化
    def deal_user_score_dict(self):
        for userid in self.begin_user_score_dict.keys():
            for artistid in self.all_artists:
                user_score_dict[user_id].setdefault(artistid,0)

        return user_score_dict

    # 获取全部的艺术家
    def get_all_artists(self):
        artists = list(set(pd.read_csv("../dataSets/data2/user_artists_train.tsv",sep='\t', header=0).values[:,1]))
        return artists

    # 优化后的倒查表方式计算物品相似度、采用惩罚活跃用户和倒查表方式计算物品相似度方法
    def itemSimilarityBest(self):
        # 得到每个用户user 评价过哪些item
        user_items = dict()
        # u 用户 items物品名称
        for u, items_with_score in self.user_score_dict.items():
            user_items.setdefault(u, set())
            for item, score in items_with_score.items():
                if score > 0:
                    user_items[u].add(item)

        # 构建倒排表
        # C 为倒排表，值代表 物品A 跟 物品B 共同被几个用户评价过
        C = dict()
        # N 记录 该物品被几个用户评价过
        N = dict()
        for u, items in user_items.items():
            for item in items:
                N.setdefault(item, 0)
                N[item] += 1
                C.setdefault(item, {})
                for itemB in items:
                    C[item].setdefault(itemB, 0)
                    if item == itemB:
                        continue
                    # 采用惩罚活跃用户
                    C[item][itemB] += 1 / math.log(1 + len(items))
        # print("C:",C)
        # print("N:",N)

        # 构建物品与物品之间的相似矩阵
        W = dict()
        for item, related_items in C.items():
            W.setdefault(item, {})
            for itemB, cuv in related_items.items():
                if item == itemB:
                    continue
                W[item].setdefault(itemB, 0.0)
                W[item][itemB] = cuv / math.sqrt(N[item] * N[itemB])
        return W

    # user 对物品 itemA 感兴趣分数
    def preItemUserScore(self, user, itemA):
        scorce = 0.0
        # self.items_sim W表 物品之间的相似度
        '''用户user 对itemA 的感兴趣分数为：和用户历史上感兴趣的物品B * 物品A与物品B的相似度'''
        for itemB, scoreB in self.user_score_dict[user].items():
            self.items_sim[itemA].setdefault(itemB, 0)
            if itemA != itemB:
                scorce += scoreB * self.items_sim[itemA][itemB]

        return scorce

    # 为商品推荐用户
    def recommend(self, userA):
        # 计算用户可能对itemA的评分
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            # 这里寻找userA 没有接触过的物品
            if self.user_score_dict[userA][item] == 0:
                # 预测用户 userA 对item 感兴趣分数
                user_item_score_dict[item] = self.preItemUserScore(userA, item)
        return dict(sorted(user_item_score_dict.items(), key=lambda d: d[1], reverse=True))


if __name__ == '__main__':
    # 获取想要预测用户的列表
    prv_userIds = list(pd.read_csv("../dataSets/data2/user_artists_test.tsv",sep='\t', header=0).values[:,0])
    train_data = get_train_data()

    ub = ItemCF(user_score_dict=train_data)

    pev = []
    for user_id in prv_userIds:
        cnt = []
        cnt.append(user_id)
        print(ub.recommend(user_id))