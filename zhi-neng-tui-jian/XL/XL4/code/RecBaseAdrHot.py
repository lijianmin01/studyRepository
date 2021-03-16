import pandas as pd

# 3、使用hotel-mess数据集实现基于地点的推荐，并附代码！
class RecBaseAdrHot:
    # file_path 数据文件的路径
    # place 地点
    # type 不同的计算方法进行排序
    # k 推荐的数目
    def __init__(self,file_path,place,type="score",k=10,sort_flag=False):
        self.file_path,self.place = file_path,place
        self.type = type
        self.k = k
        self.sort = sort_flag
        # 加载数据
        self.data = self.load_data()

    # 加载数据函数
    def load_data(self):
        dataSet = pd.read_csv(self.file_path, header=0, sep=",", encoding="gbk")
        return dataSet[dataSet["addr"]==self.place]

    # 根据不同的方式，对特定的地点place 进行推荐
    def reccomond(self):
        if self.type in ["score","lowest_price", "comment_num", "open_time","decoration_time"]:
            # 根据type 对数据进行排序
            dataSet = self.data.sort_values(by=[self.type, "lowest_price"], ascending=self.sort)[:self.k]
            # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换
            return dict(dataSet.filter(items=["name", self.type]).values)
        # 综合排序,根据不同因素所占权重进行排序
        elif self.type == "combine":
            # 过滤数据，得到所要使用的数据
            dataSet = self.data.filter(items=["name", "score", "comment_num", "lowest_price", "decoration_time", "open_time", "lowset_price"])

            # 对店面装饰时间进行处理
            dataSet["decoration_time"] = dataSet["decoration_time"].apply(lambda x: int(x) - 2018)
            # 对店面开店时间进行处理
            dataSet["open_time"] = dataSet["open_time"].apply(lambda x: 2018 - int(x))

            # 对数据进行归一化处理 （data - min）/(max - min)
            for shop in dataSet.keys():
                # 排除标题行
                if shop != "name":
                    dataSet[shop] = (dataSet[shop] - dataSet[shop].min()) / (dataSet[shop].max() - dataSet[shop].min())

            # 这里认为评分权重为1 评论数目权重为2 装修和开业时间权重为0.5
            dataSet[self.type] = 1 * dataSet["score"] + 2 * dataSet["comment_num"] + 0.5 * dataSet["decoration_time"] + 0.5 * dataSet["open_time"]

            return dict(dataSet.filter(items=["name", self.type]).values)




if __name__ == "__main__":
    path = "../hotel-mess/hotel-mess.csv"

    hotel_rec = RecBaseAdrHot(path, place="丰台区", type="combine", k=10, sort_flag=False)
    results = hotel_rec.reccomond()
    print(results)