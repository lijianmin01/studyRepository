# 导入相关的包
import os, math
# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split


# 获取数据
class RecommedFrieds:
    '''
    data_path: nodeId.edges 文件的路径
        nodeId.edges ： 里面的数据代表 nodeId 指向的所有数据   每一行A  B   ==>  A->B  nodeId->A  nodeId->B
        method ： 不同的计算用户相似度的方法
    '''

    def __init__(self, method=1,data_name_list=None):
        self.data_name_list = data_name_list
        self.data_path_list = self.get_data_path()
        self.train,self.test = self.split_data()
        self.data_out = self.get_user_out()
        self.data_in = self.get_user_in()

        self.method = method

        if method == 1:
            self.user_sim = self.userSimilarityMethod1()
        elif method == 2:
            self.user_sim = self.userSimilarityMethod2()
        elif method == 3:
            self.user_sim = self.userSimilarityMethod3()

    # 获取个个文件的路径
    def get_data_path(self):
        # nodeId.edges 文件路径
        data_path = "/home/lijianmin/github/github_not_data/twitter/"

        # data_name_list = ['12831.edges', '163374693.edges']

        data_path_list = []

        for data_name in self.data_name_list:
            data_path_list.append(data_path + data_name)

        return data_path_list

    # 讲数据分成训练集和测试集
    def split_data(self):
        dataSet = list()
        for d_p in self.data_path_list:
            with open(d_p, "r") as f:
                dataSet.extend(f.readlines())

        train, test = train_test_split(dataSet,test_size=0.1,random_state=16)
        return train,test

    # Wout ( u , v ) 公式中 out(u) 是在社交网络图中用户 u 指向的其他好友的集合。
    def get_user_out(self):
        user_out = dict()

        filename = self.data_name_list[0].split(".")[0]
        user_out.setdefault(filename,set())
        for line in self.train:
            (user_A, user_B) = line.split()
            user_out[filename].add(user_A)
            user_out[filename].add(user_B)
            # A -> B
            user_out.setdefault(user_A, set())
            user_out[user_A].add(user_B)

        return user_out

    # in(u)  是在社交网络图中指向用户 u 的用户的集合
    def get_user_in(self):
        user_in = dict()

        filename = self.data_name_list[0].split(".")[0]
        for line in self.train:
            (user_A, user_B) = line.split()
            user_in.setdefault(user_A, set())
            user_in.setdefault(user_B, set())
            user_in[user_A].add(filename)
            user_in[user_B].add(filename)
            # A -> B
            user_in[user_B].add(user_A)

        return user_in


    # W out (u,v) = |out(u) & out(v)| / sqrt(out(u) * out(v))
    def userSimilarityMethod1(self):
        user_sim = {}
        # 用户 A
        for user_A in self.data_out.keys():
            # 用户B
            for user_B in self.data_out.keys():
                if user_A == user_B:
                    continue
                user_sim.setdefault(user_A, {}).setdefault(user_B, 0.0)
                user_sim[user_A][user_B] = 1.0 * (len(self.data_out[user_A] & self.data_out[user_B])) / math.sqrt(len(self.data_out[user_A]) * len(self.data_out[user_B]))

        return user_sim

    # W in (u,v) = |in(u) & in(v)| / sqrt(in(u) * in(v))
    def userSimilarityMethod2(self):
        user_sim = {}
        # 用户 A
        for user_A in self.data_in.keys():
            # 用户B
            for user_B in self.data_in.keys():
                user_sim.setdefault(user_A, {}).setdefault(user_B, 0.0)
                if user_A == user_B:
                    continue

                user_sim[user_A][user_B] = 1.0 * (len(self.data_in[user_A] & self.data_in[user_B])) / math.sqrt(len(self.data_in[user_A]) * len(self.data_in[user_B]))

        return user_sim



    # W out,in (u,v) = |out(u) & in(v)| / sqrt(out(u) * in(v))
    def userSimilarityMethod3(self):
        user_sim = {}
        # 用户 A
        for user_A in self.data_out.keys():
            # 用户B
            for user_B in self.data_out.keys():
                if user_A == user_B:
                    continue
                user_sim.setdefault(user_A, dict()).setdefault(user_B, 0.0)
                if (user_A in self.data_in) and (user_B in self.data_in) and (user_A in self.data_out) and (user_B in self.data_out):
                    user_sim[user_A][user_B] = 1.0 * (len(self.data_out[user_A] & self.data_in[user_B])) / math.sqrt(
                        len(self.data_out[user_A]) * len(self.data_in[user_B]))

        return user_sim


    def UserRecommedFrieds(self, userA):
        # 用户A 不认识的朋友排名
        rank = {}

        for related_user,rate in self.user_sim[userA].items():
            if rate>0:
                for userC,c_rate in self.user_sim[related_user].items():
                    if c_rate>0:
                        rank.setdefault(userC, 0.0)
                        # if userA!=userC and (userC not in self.user_sim[userA].keys()):
                        if userA != userC:
                            rank[userC]+=rate*c_rate

        # 排除与userA 关注的人（既A 认识的人）
        for userA_firend in self.data_out[userA]:
            rank[userA_firend] = 0

        # print("#",self.user_sim[userA])
        rank = dict(sorted(rank.items(), key=lambda d: d[1], reverse=True)[:20])
        return rank


    # 验证程序，对各种计算方法进行打分
    def review(self):
        test_sum = len(self.test)
        prv_right = 0
        for line in self.test:
            (userA, userB) = line.split()
            try:
                prv_users = self.UserRecommedFrieds(userA).keys()
                if userB in prv_users:
                    prv_right+=1
            except:
                # userA 没有出现在训练集中，跳过
                continue

        print("预测正确率为：{}".format(prv_right*1.0/test_sum))






if __name__ == '__main__':
    #data_name_list = ['12831.edges', '163374693.edges']
    data_name_list = ['12831.edges']
    # 为用户 1186 推荐新朋友
    userA = "19101100"
    print("采用不同的方式为用户 {} 推荐好友：".format(userA))
    ways = [str(i) for i in range(1, 4)]
    #ways = ['2']

    test = []
    for way in ways:
        print("方式 {} :".format(way))
        rf = RecommedFrieds(method=int(way),data_name_list=data_name_list)
        print("该预测方式，在测试集上的正确率为：", end=" ")
        rf.review()
        rank = rf.UserRecommedFrieds(userA)
        print("推荐的好友为：", rank.keys())
        test.append(rank.keys())








