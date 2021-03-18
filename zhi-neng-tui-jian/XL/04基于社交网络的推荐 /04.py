# 导入相关的包
import os

# 获取数据
class RecommedFrieds:
    '''
    data_path: nodeId.edges 文件的路径
        nodeId.edges ： 里面的数据代表 nodeId 指向的所有数据   每一行A  B   ==>  A->B  nodeId->A  nodeId->B
        method ： 不同的计算用户相似度的方法
    '''
    def __init__(self,data_path,method=1):
        self.data_path = data_path
        self.data_out = self.get_user_out()
        self.data_in = self.get_user_in()

        if method == 1:
            self.user_sim = self.userSimilarityMethod1()
        elif method == 2:
            self.user_sim = self.userSimilarityMethod2()
        elif method == 3:
            self.user_sim = self.userSimilarityMethod3()

    # Wout ( u , v ) 公式中 out(u) 是在社交网络图中用户 u 指向的其他好友的集合。
    def get_user_out(self):
        (path,filename) = os.path.split(self.data_path)
        (filename,hz) = filename.split(".")
        print(filename,hz)

    # in(u)  是在社交网络图中指向用户 u 的用户的集合
    def get_user_in(self):

        pass

    def userSimilarityMethod1(self):

        pass

    def userSimilarityMethod2(self):

        pass

    def userSimilarityMethod3(self):

        pass


if __name__ == '__main__':
    # nodeId.edges 文件路径
    data_path = r"/home/lijianmin/github/github_not_data/twitter/12831.edges"
    rf = RecommedFrieds(data_path)
    rf.get_user_out()

    pass
