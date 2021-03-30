import random
import math

from sklearn.model_selection import train_test_split


class BiasFLM():
    def __init__(self,data_path,F=8,alpha=0.1,lmbd=0.1,max_iter=2):
        self.data_path=data_path
        self.F=F
        self.alpha=alpha
        self.lmbd=lmbd
        self.max_iter=max_iter
        self.train_dict, self.test_dict = self.loadData()
        self.P,self.Q,self.bu,self.bi,self.mu,self.movies=self.init_param()


    def init_param(self):
        P=dict()
        Q=dict()
        bu=dict()
        bi=dict()
        cnt=0
        mu=0
        #存放所有电影
        movies=list()
        for user,items in self.train_dict.items():
            P[user] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
            bu[user] = 0
            cnt+=len(items)
            for item,record in items.items():
                movies.append(item)
                mu+=record
                if item not in Q:
                    Q[item] = [random.random() / math.sqrt(self.F) for x in range(self.F)]
                bi[item]=0
        mu/=cnt
        return P,Q,bu,bi,mu,movies
    def loadData(self):
        data=[]
        with open(self.data_path,'r') as f:
            lines=f.readlines()
        for line in lines:
            userid,itemid,record,timestamp=line.split("::")
            data.append((userid, itemid, int(record)))

        train_list, test_list = train_test_split(data, test_size=0.1, random_state=1234)
        train_dict=self.transform(train_list)
        test_dict=self.transform(test_list)
        return train_dict,test_dict
    def transform(self,data):
        data_dict=dict()
        for userid,itemid,record in data:
            data_dict.setdefault(userid,dict())
            data_dict[userid].setdefault(itemid,record)
        return data_dict

    def train(self):
        for step in range(self.max_iter):
            for user,items in self.train_dict.items():
                for item,rui in items.items():
                    hat_ui=self.predict(user,item)
                    err_ui=rui-hat_ui

                    self.bu[user] += self.alpha * (err_ui - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd * self.bi[item])
                    for f in range(self.F):
                        self.P[user][f]+=self.alpha*(err_ui*self.Q[item][f]-self.lmbd*self.P[user][f])
                        self.Q[item][f]+=self.alpha*(err_ui*self.P[user][f]-self.lmbd*self.Q[item][f])
            self.alpha*=0.9
            print(self.P)
            print(self.Q)

    def predict(self,user,item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F)) + self.bu[user] + self.bi[item] + self.mu

    def recommand(self,user,N=10):
        rank=dict()
        interested_movies=self.train_dict[user].keys()
        for movie in self.movies:
            if movie not in interested_movies:
                rank[movie]=self.predict(user,movie)

        temp= sorted(rank.items(),key=lambda x:x[1],reverse=True)[:N]
        recommand_list=[key[0] for key in temp]
        return recommand_list
    def percision(self,N=10):
        self.train()
        hit=0
        num=0
        for user in self.test_dict.keys():
            recommand_list=self.recommand(user,N)
            for i in recommand_list:
                if i in self.test_dict[user].keys():
                    hit+=1
            num+=N
        return hit/num
if __name__ == '__main__':
    biasflm=BiasFLM('ratings.dat')
    print(biasflm.percision())