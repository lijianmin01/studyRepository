# 导入sklearn 用来划分数据
from sklearn.model_selection import train_test_split

data_path = "/home/lijianmin/github/github_not_data/twitter/"

data_name_list = ['12831.edges','163374693.edges']

data_path_list = []

for data_name in data_name_list:
    data_path_list.append(data_path+data_name)


dataSet = list()
for d_p in data_path_list:
    with open(d_p,"r") as f:
        dataSet.extend(f.readlines())


train,test = train_test_split(dataSet,random_state=13,test_size=0.1)
print(train)
print(test)


