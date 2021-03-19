# 导入相关的包
import os, math


# 获取数据
class RecommedFrieds:
    '''
    data_path: nodeId.edges 文件的路径
        nodeId.edges ： 里面的数据代表 nodeId 指向的所有数据   每一行A  B   ==>  A->B  nodeId->A  nodeId->B
        method ： 不同的计算用户相似度的方法
    '''

    def __init__(self, data_path, method=1):
        self.data_path = data_path
        self.data_out = self.get_user_out()
        self.data_in = self.get_user_in()

        self.method = method

        if method == 1:
            self.user_sim = self.userSimilarityMethod1()
        elif method == 2:
            self.user_sim = self.userSimilarityMethod2()
        elif method == 3:
            self.user_sim = self.userSimilarityMethod3()

    # Wout ( u , v ) 公式中 out(u) 是在社交网络图中用户 u 指向的其他好友的集合。
    def get_user_out(self):
        (path, filename) = os.path.split(self.data_path)
        (filename, hz) = filename.split(".")
        user_out = dict()
        user_out.setdefault(filename, set())
        # 打开文件，遍历每一行数据
        with open(self.data_path, "r") as f:
            for line in f.readlines():
                (user_A, user_B) = line.split()
                user_out[filename].add(user_A)
                user_out[filename].add(user_B)
                # A -> B
                user_out.setdefault(user_A, set())
                user_out[user_A].add(user_B)

        return user_out

    # in(u)  是在社交网络图中指向用户 u 的用户的集合
    def get_user_in(self):
        (path, filename) = os.path.split(self.data_path)
        (filename, hz) = filename.split(".")
        user_in = dict()
        # 打开文件，遍历每一行数据
        with open(self.data_path, "r") as f:
            for line in f.readlines():
                (user_A, user_B) = line.split()
                user_in.setdefault(user_A, set())
                user_in.setdefault(user_B, set())
                user_in[user_A].add(filename)
                user_in[user_B].add(filename)
                # A -> B
                user_in[user_B].add(user_A)

        # print(user_in)
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
                user_sim[user_A][user_B] = 1.0 * (len(self.data_out[user_A] & self.data_out[user_B])) / math.sqrt(
                    len(self.data_out[user_A]) * len(self.data_out[user_B]))

        return user_sim

    # W in (u,v) = |in(u) & in(v)| / sqrt(in(u) * in(v))
    def userSimilarityMethod2(self):
        user_sim = {}
        # 用户 A
        for user_A in self.data_in.keys():
            # 用户B
            for user_B in self.data_in.keys():
                if user_A == user_B:
                    continue
                user_sim.setdefault(user_A, {}).setdefault(user_B, 0.0)
                if (user_A in self.data_in) and (user_B in self.data_in):
                    user_sim[user_A][user_B] = 1.0 * (len(self.data_in[user_A] & self.data_in[user_B])) / math.sqrt(
                        len(self.data_in[user_A]) * len(self.data_in[user_B]))
                    #print(user_sim[user_A][user_B])

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
                user_sim.setdefault(user_A, {}).setdefault(user_B, 0.0)
                if (user_A in self.data_in) and (user_B in self.data_in) and (user_A in self.data_out) and (
                        user_B in self.data_out):
                    user_sim[user_A][user_B] = 1.0 * (len(self.data_out[user_A] & self.data_in[user_B])) / math.sqrt(
                        len(self.data_out[user_A]) * len(self.data_in[user_B]))

        return user_sim

    # 为用户A 推荐好友
    def UserRecommedFrieds(self, userA):
        # 用户A 不认识的朋友排名
        new_firends_rank = dict()
        # 寻找A 不认的朋友B
        for userB in self.user_sim[userA].keys():
            if self.user_sim[userA][userB] == 0.0:
                new_firends_rank.setdefault(userB, 0.0)

        for new_firend in new_firends_rank.keys():
            # A 认识的朋友 C &&  C 认识 new_firend
            for userC in self.user_sim[userA].keys():
                if self.user_sim[userA][userC] > 0.0 and self.user_sim[userC][new_firend] > 0.0:
                    new_firends_rank[new_firend] += self.user_sim[userA][userC] * self.user_sim[userC][new_firend]

        rank = dict(sorted(new_firends_rank.items(), key=lambda d: d[1], reverse=True))

        return rank


# # 为用户推荐10名新朋友
def get_nearest_new_firends(rank):
    user_ids = rank.keys()
    if len(user_ids) < 10:
        for user_id in user_ids:
            if rank[user_id] > 0.0:
                print("用户：{}  评分：{}".format(user_id, rank[user_id]))
            else:
                break
    else:
        cnt = 0
        for user_id in user_ids:
            if rank[user_id] > 0.0:
                print("用户：{}  评分：{}".format(user_id, rank[user_id]))
            else:
                break
            cnt += 1
            if cnt > 10:
                break


if __name__ == '__main__':
    # nodeId.edges 文件路径
    data_path = r"/home/lijianmin/github/github_not_data/twitter/78813.edges"

    # 为用户 1186 推荐新朋友
    userA = "627363"
    print("采用不同的方式为用户 {} 推荐好友：".format(userA))
    ways = [str(i) for i in range(1, 4)]
    # ways = ['2']
    for way in ways:
        print("方式 {} :".format(way))
        rf = RecommedFrieds(data_path, method=int(way))
        rank = rf.UserRecommedFrieds(userA)
        get_nearest_new_firends(rank)

        # print(len(rf.data_out['12831']))
        # print(rf.data_in)
        # print(rf.data_in['398874773'])
        # print(rf.data_in['12831'])



