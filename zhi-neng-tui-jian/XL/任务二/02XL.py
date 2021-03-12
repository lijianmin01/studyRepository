import random
import math

# 生成数据
# user_nums 用户的个数  item_nums 商品的个数
def create_data(user_nums,item_nums):
    # 用户名称列表
    Users = list(map(lambda x:(chr(x)), range(ord('A'), ord('A') + user_nums)))
    # 商品名称列表
    Items = list(map(lambda x:(chr(x)), range(ord('a'),ord('a')+item_nums)))
    # 用户打分选项
    score_dict = [0.0,2.0,3.0,3.5,4.0,0.0,5.0,0.0]

    user_score_dict = dict()

    for user in Users:
        user_score_dict.setdefault(user,{})
        for item in Items:
            user_score_dict[user].setdefault(item,0.0)
            # 生成随机分数
            random_sorce = score_dict[random.randint(0,7)]
            user_score_dict[user][item]=random_sorce
    
    return user_score_dict

class UserCF:

    def __init__(self,user_score_dict,way=1):
        self.user_score_dict = user_score_dict
        if way == 1:
            self.users_sim = self.userSimilarity()
        elif way == 2:
            self.users_sim = self.userSimilarityBetter()
        elif way == 3:
            self.users_sim = self.userSimilarityBest()


    # 计算两两用户之间的相似度
    def userSimilarity(self):
        W = dict()
        for u in self.user_score_dict.keys():
            W.setdefault(u, {})
            for v in self.user_score_dict.keys():
                if u == v:
                    continue
                u_set = set([k for k in self.user_score_dict[u].keys() if self.user_score_dict[u][k] > 0.0])
                v_set = set([k for k in self.user_score_dict[u].keys() if self.user_score_dict[v][k] > 0.0])
                W[u][v] = float(len(u_set & v_set)) / math.sqrt(len(u_set) * len(v_set))
        return W

    # 用户之间的相似度，采用优化算法时间复杂度的方法
    def userSimilarityBetter(self):
        # 得到每个item 被那些user评价过
        item_users = dict()
        # u 用户   items 物品的名称
        for u, items in self.user_score_dict.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                if self.user_score_dict[u][i] > 0:
                    item_users[i].add(u)

        # 构建倒排表
        # C 为倒排表 数字代表 用户1 跟 用户 2 共同浏览过几种产品
        C = dict()
        # N 记录 用户跟浏览过几种产品
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if u == v:
                        continue
                    C[u][v] += 1

        # print("C:", C)
        # print("N:", N)
        # 构建相似度矩阵
        W = dict()

        for u, related_users in C.items():
            W.setdefault(u, {})
            for v, cuv in related_users.items():
                if u == v:
                    continue
                W[u].setdefault(v, 0.0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])

        return W

    # 惩罚re计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的算法
    def userSimilarityBest(self):
        # 得到每个item 被那些user评价过
        item_users = dict()
        for u, items in self.user_score_dict.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                if self.user_score_dict[u][i] > 0:
                    item_users[i].add(u)

        # 构建倒排表
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if u == v:
                        continue
                    # 在产品item 用户 u 跟 用户 v 的 倒排表 数值
                    C[u][v] += 1 / math.log(1 + len(users))

        # print("C:", C)
        # print("N:", N)
        # 构建相似度矩阵
        W = dict()
        # u 是 用户 在一列上跟u相关用户（包括u）
        for u, related_users in C.items():
            W.setdefault(u, {})
            # v代表一个用户  u代表用户  cuv代表数值
            for v, cuv in related_users.items():
                if u == v:
                    continue
                W[u].setdefault(v, 0.0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])

        return W

    # 预测用户对item的评分
    def preUserItemScore(self, userA, item):
        scorce = 0.0
        '''self.users_sim 即求出的W 表 各个用户之间的相似度'''
        for user in self.users_sim[userA].keys():
            if user != userA:
                '''用户userA对item的分数为：userA 与 user 相似度   *  用户user对item的评分'''
                scorce += self.users_sim[userA][user] * self.user_score_dict[user][item]

        return scorce

    # 为用户推荐物品
    def recommend(self, userA):
        # 计算userA,对item可能的评分
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            # 这里寻找userA 没有接触过的物品
            if self.user_score_dict[userA][item] == 0:
                # 预测用户 userA 对item 感兴趣分数
                user_item_score_dict[item] = self.preUserItemScore(userA, item)
        # 这里修改以下 返回分数从大到小排序
        # {'a': 2.8577380332470415, 'c': 1.8371173070873839, 'd': 4.286607049870562}
        # return user_item_score_dict
        return dict(sorted(user_item_score_dict.items(), key=lambda d: d[1], reverse=True))


class ItemCF:
    def __init__(self,user_score_dict,way=1):
        self.user_score_dict = user_score_dict
        if way == 1:
            self.items_sim = self.itemSimilarity()
        elif way == 2:
            self.items_sim = self.itemSimilarityBetter()
        elif way == 3:
            self.items_sim = self.itemSimilarityBest()


    # 两两物品之间计算、
    def itemSimilarity(self):
        # W 用来记录物品与物品之间的相似矩阵
        W = dict()
        # 得到每个item 被那些user评价过
        item_users = dict()
        for u, items in self.user_score_dict.items():
            for item, scorce in items.items():
                item_users.setdefault(item, set())
                if scorce > 0:
                    item_users[item].add(u)

        # 来计算物品与物品之间的相似度
        for itemA in item_users.keys():
            W.setdefault(itemA, {})
            for itemB in item_users.keys():
                if itemA == itemB:
                    continue
                W[itemA][itemB] = float(len(item_users[itemA] & item_users[itemB]) / math.sqrt(
                    len(item_users[itemA]) * len(item_users[itemB])) * 1.0)

        return W

    # 优化后的倒查表方式计算物品相似度
    def itemSimilarityBetter(self):
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
                    C[item][itemB] += 1
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
    # 生成8名用户的名字
    users = list(map(lambda x: (chr(x)), range(ord('A'), ord('A') + 8)))
    # 每一种算法的三种不同实现方式，标号
    ways = [i for i in range(1, 4)]
    print("随机生成数据 8名用户  6件物品  随机打分，生成的数据为：")

    user_score_dict = create_data(8,6)
    for u,items in user_score_dict.items():
        print(u,items)

    
    print("基于用户的协同过滤(UserCF)算法的实现:")
    for user in users:
        print("UserCF 用户{}推荐的商品：".format(user))
        for way in ways:
            print("   方式{}：".format(way), end=" ")
            ub = UserCF(user_score_dict=user_score_dict,way=way)
            print(ub.recommend(user))


    print("\n\n基于物品的协同过滤(ItemCF)算法的实现:")
    for user in users:
        print("ItemCF 用户{}推荐的商品：".format(user))
        for way in ways:
            print("   方式{}：".format(way), end=" ")
            ub = ItemCF(user_score_dict=user_score_dict,way=way)
            print(ub.recommend(user))




