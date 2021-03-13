#
#
# # 调用sklearn 库生成数据的函数
# def create_data():
#     from sklearn.datasets import make_classification
#     data,target = make_classification(n_samples=100,        # 样本个数
#                                n_features=2,          # 特征个数
#                                n_informative=2,        # 有效特征个数
#                                n_redundant=0,          # 冗余特征个数（有效特征的随机组合）
#                                n_repeated=0,           # 重复特征个数（有效特征和冗余特征的随机组合）
#                                n_classes=2,            # 样本类别
#                                n_clusters_per_class=1, # 簇的个数
#                                random_state=0)
#     return data,target
#
# # 生成数据的可视化函数
# def show_data(data,target):
#     import matplotlib.pyplot as plt
#     data_label = list(set(target))
#     plt.figure()
#     for i in range(len(target)):
#         if(target[i]==data_label[0]):
#             plt.scatter(data[i][0],data[i][1],c='r')
#         else:
#             plt.scatter(data[i][0],data[i][1],c='g')
#     plt.show()
#
# # 显示数据的分布情况，以及显示划分效果
# def show_data_line(data,target,w,b):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     data_label = list(set(target))
#     plt.figure()
#     for i in range(len(target)):
#         if(target[i]==data_label[0]):
#             plt.scatter(data[i][0],data[i][1],c='r')
#         else:
#             plt.scatter(data[i][0],data[i][1],c='g')
#     # 为了更客观的看出图像的变化，这里我们先找出原来的图像的坐标轴的横纵坐标的最大，最小
#     # x_min,x_max,y_min,y_max = 0.0,0.0,0.0,0.0
#
#     [x_min,y_min] = np.min(data,axis=0)
#     [x_max,y_max] = np.max(data,axis=0)
#
#
#     XX=np.linspace(x_min-1,x_max+1)
#     YY=-(w[0]/w[1])*XX-b/w[1]
#     plt.plot(XX,YY,'-c',label='Hyperplane')
#     plt.show()
#
# if __name__=='__main__':
#     # 生成数据
#     data,target = create_data()
#     # 因为这里生成数据标签为0 1 所以我们修改为 -1 1
#     for i in range(len(target)):
#         if target[i]==0:
#             target[i]=-1
#
#     # 对生成数据的可视化
#     show_data(data,target)
#
#     # 开始模拟感知机的训练过程
#     # 初始化 w , b
#     w = np.array((0.0,0.0))
#     b = 0.0
#
#     # 所有数据训练10轮
#     num_epochs = 1
#
#     # 每50条数据为一个周期
#     epoch = 20
#
#     # 学习率 0.01
#     lr = 0.01
#     import sklearn
#     while(num_epochs<=10):
#         # 随机打乱数据
#         data,target = sklearn.utils.shuffle(data,target)
#         for i in range(len(target)):
#