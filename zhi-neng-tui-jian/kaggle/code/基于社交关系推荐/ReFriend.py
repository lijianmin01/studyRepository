import os,math

import numpy as np
import pandas as pd

class RecommedFrieds:

    def __init__(self):
        self.data_out,self.data_in = self.get_user_out_and_in()
        self.all_user_id = self.get_all_userid()
        self.all_artist_id = self.get_all_artistid()
        self.user_sim = self.userSimilarity()

        self.user_artist_score = self.get_train_data()

    # 获取训练数据
    def get_train_data(self):
        # 读取数据
        train = pd.read_csv("../../dataSets/data2/user_artists_train.tsv", sep='\t', header=0).values

        userIDs = list(set(train[:, 0]))

        # 这里因为每个用户的时间不一样，所以占用听歌时间的比例也不一样，所以这里我们用听歌所占比例来代表权重
        # 用来记录用户的听歌的总次数
        user_artists_train_sum = {}

        # 初始化用户的听歌次数
        for user_id in userIDs:
            user_artists_train_sum[user_id] = 0.0

        # 开始记录用户的听歌总次数
        for i in range(len(train)):
            user_artists_train_sum[train[i][0]] += train[i][2]

        # print(user_artists_train_sum)
        # 计算每位艺术家在用户的听歌比例
        new_user_artists_train = {}
        for i in range(len(train)):
            new_user_artists_train.setdefault(train[i][0], {})
            new_user_artists_train[train[i][0]].setdefault(train[i][1], 0.0)
            new_user_artists_train[train[i][0]][train[i][1]] = 1.0 * train[i][2] / user_artists_train_sum[train[i][0]] * 100

        return new_user_artists_train

    def get_user_out_and_in(self):
        user_out = dict()
        user_in = dict()
        userid_friendid = pd.read_csv("/home/lijianmin/github/studyRepository/zhi-neng-tui-jian/kaggle/dataSets/data2/user_friends.tsv", sep='\t', header=0).values

        for i in range(len(userid_friendid)):
            user_id = userid_friendid[i,0]
            friend_id = userid_friendid[i,1]

            user_out.setdefault(user_id,set())
            user_out[user_id].add(friend_id)

            user_in.setdefault(friend_id,set())
            user_in[friend_id].add(user_id)

        return user_out,user_in

    def userSimilarity(self):
        user_sim = {}
        for user_A in self.all_user_id:
            for user_B in self.all_user_id:
                user_sim.setdefault(user_A, dict())
                user_sim[user_A].setdefault(user_B, 0.0)
                if user_A == user_B:
                    continue
                user_sim[user_A][user_B] = 1.0*(len(self.data_out[user_A] & self.data_in[user_B])) / (math.sqrt(len(self.data_out[user_A]) * len(self.data_in[user_B])))

        return user_sim

    def get_all_userid(self):
        return list(pd.read_csv("/home/lijianmin/github/studyRepository/zhi-neng-tui-jian/kaggle/dataSets/data2/user_friends.tsv", sep='\t',header=0).values[:,0])

    # 为用户A推荐物品  ：  物品分数= 用户A B之间的相似度  *  用户B 对该商品所打的分
    def get_all_artistid(self):
        return list(set(pd.read_csv("../../dataSets/data2/user_artists_train.tsv", sep='\t', header=0).values[:, 1]))

    def recommed(self,userA):
        # 首先计算用户 与 那一些艺术家发生过关系
        # 每个用户接触过那一些artists
        user_artists = dict()
        for userid in self.user_artist_score.keys():
            user_artists.setdefault(userid, set())
            for artist, score in self.user_artist_score[userid].items():
                if score > 0:
                    user_artists[userid].add(artist)

        arts_score = dict()
        for art in self.all_artist_id:
            if art not in user_artists[userA]:
                arts_score.setdefault(art,0.0)
                for userB in self.all_user_id:
                    if (userA != userB) and (art in user_artists[userB]):
                        arts_score[art] += self.user_sim[userA][userB] * self.user_artist_score[userB][art]

        return list(dict(sorted(arts_score.items(), key=lambda d: d[1], reverse=True)).keys())[:5]


def zhuanhua(l):
    return str(l[0]) + ' ' + str(l[1]) + ' ' + str(l[2]) + ' ' + str(l[3]) + ' ' + str(l[4])


# 预测结果并生成数据
def prv_run():
    # 要预测的用户
    prv_userIds = list(pd.read_csv("../../dataSets/data2/user_artists_test.tsv", sep='\t', header=0).values[:, 0])

    ub = RecommedFrieds()

    print("ok")
    flag = 0

    result = []
    for i in range(len(prv_userIds)):
        userid = prv_userIds[i]
        cnt = []
        cnt.append(userid)
        temp = ub.recommed(userid)
        arts = zhuanhua(temp)
        cnt.append(arts)
        print(userid,arts)
        result.append(cnt)

        if ((i + 1) % 10 == 0):
            df = pd.DataFrame(np.array(result), columns=['userID', 'artistIDs'])
            df.to_csv("temp/{}.csv".format(flag), index=False)
            print("{}已完成".format(flag))
            flag += 1
            result = []

    df = pd.DataFrame(np.array(result), columns=['userID', 'artistIDs'])
    df.to_csv("temp/{}.csv".format(flag), index=False)
    print("{}已完成".format(flag))


if __name__ == '__main__':
    prv_run()
