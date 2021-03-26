# ItemCF算法

import os,math
import pandas as pd
import numpy as np

class ItemCF:

    def __init__(self):
        self.user_artist_score = self.get_train_data()
        
        self.items_sim = self.ItemSimilarity()

        self.all_art = self.get_all_art()
        
    # 获取训练数据
    def get_train_data(self):
        # 读取数据
        train = pd.read_csv("../dataSets/data2/user_artists_train.tsv", sep='\t', header=0).values

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

    def ItemSimilarity(self):
        # 每个用户接触过那一些artists
        user_artists = dict()
        for userid in self.user_artist_score.keys():
            user_artists.setdefault(userid,set())
            for artist,score in self.user_artist_score[userid].items():
                if score>0:
                    user_artists[userid].add(artist)

        # 构建倒排表
        # C 为倒排表，带表物品A 跟物品B 共同被几个用户评价
        C = dict()
        # N 记录 艺术家 与 多少用户有关系
        N = dict()

        for userid,artists in user_artists.items():
            for artist in artists:
                N.setdefault(artist,0)
                N[artist]+=1
                C.setdefault(artist,{})
                for artistB in artists:
                    C[artist].setdefault(artistB,0)
                    if artist == artistB:
                        continue
                    # 惩罚活跃用户
                    C[artist][artistB]+= 1.0 / math.log(1+len(artists))


        # 构建艺术家与艺术家之间的相似矩阵
        W = dict()
        for artist,related_artists in C.items():
            W.setdefault(artist,{})
            for artistB,cuv in related_artists.items():
                if artist == artistB:
                    continue
                W[artist].setdefault(artistB,0.0)
                W[artist][artistB] = cuv/math.sqrt(N[artist]*N[artistB]*1.0)

        return W

    # user 对 艺术家A 的感兴趣分数
    def preArtistuserScore(self,user,artA):
        scorce = 0.0

        # 艺术家之间的相似度
        for artB,scoreB in self.user_artist_score[user].items():
            self.items_sim[artA].setdefault(artB,0)
            if artA!=artB:
                scorce += scoreB * self.items_sim[artA][artB]

        return scorce

    def get_all_art(self):
        artists = list(set(pd.read_csv("../dataSets/data2/user_artists_train.tsv", sep='\t', header=0).values[:, 1]))
        return artists

    #为用户推荐art
    def recommend(self,userA):
        # 计算用户可能对艺术家的评价
        user_art_score_dict = dict()
        for art in self.all_art:
            if art not in self.user_artist_score[userA].keys():
                user_art_score_dict[art] = self.preArtistuserScore(userA,art)

        return list(dict(sorted(user_art_score_dict.items(), key=lambda d: d[1], reverse=True)).keys())[:5]


# 预测结果并生成数据
def prv_run1(name="test"):

    result = []
    ub = ItemCF()

    # 要预测的用户
    prv_userIds = list(pd.read_csv("../dataSets/data2/user_artists_test.tsv", sep='\t', header=0).values[:, 0])

    for userid in prv_userIds:
        print(userid)
        cnt = []
        cnt.append(userid)
        cnt.append(ub.recommend(userid))
        result.append(cnt)

    df = pd.DataFrame(np.array(result),columns=['userID','artistIDs'])

    df.to_csv("{}.csv".format(name),index=False)


def zhuanhua(l):
    return str(l[0]) + ' ' + str(l[1]) + ' ' + str(l[2]) + ' ' + str(l[3]) + ' ' + str(l[4])


# 预测结果并生成数据
def prv_run():
    # 要预测的用户
    prv_userIds = list(pd.read_csv("../dataSets/data2/user_artists_test.tsv", sep='\t', header=0).values[:, 0])

    ub = ItemCF()

    flag = 20

    result = []
    for i in range(19*50,len(prv_userIds)):
        userid = prv_userIds[i]
        print(userid)
        cnt = []
        cnt.append(userid)
        temp = ub.recommend(userid)
        cnt.append(zhuanhua(temp))
        result.append(cnt)

        if ((i + 1) % 50 == 0):
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



