import sklearn
import numpy as np

def create_data():
    from sklearn.datasets import make_classification
    data, target = make_classification(n_samples=40,  # 样本个数
                                       n_features=2,  # 特征个数
                                       n_informative=2,  # 有效特征个数
                                       n_redundant=0,  # 冗余特征个数（有效特征的随机组合）
                                       n_repeated=0,  # 重复特征个数（有效特征和冗余特征的随机组合）
                                       n_classes=2,  # 样本类别
                                       n_clusters_per_class=2,  # 簇的个数
                                       random_state=0)
    return data, target


# 数据的可视化
def show_data(data, target):
    import matplotlib.pyplot as plt
    data_label = list(set(target))
    plt.figure()
    for i in range(len(target)):
        if (target[i] == data_label[0]):
            plt.scatter(data[i][0], data[i][1], c='r')
        else:
            plt.scatter(data[i][0], data[i][1], c='g')
    plt.show()


data, target = create_data()
print(data[0], target[0])
show_data(data, target)
for i in range(len(target)):
    if target[i] == 0:
        target[i] = -1


# 分类效果展示
def show_data_line(data, target, w, b):
    import matplotlib.pyplot as plt
    import numpy as np
    data_label = list(set(target))
    plt.figure()
    for i in range(len(target)):
        if (target[i] == data_label[0]):
            plt.scatter(data[i][0], data[i][1], c='r')
        else:
            plt.scatter(data[i][0], data[i][1], c='g')

    # 为了更客观的看出图像的变化，这里我们先找出原来的图像的坐标轴的横纵坐标的最大，最小
    # x_min,x_max,y_min,y_max = 0.0,0.0,0.0,0.0

    [x_min, y_min] = np.min(data, axis=0)
    [x_max, y_max] = np.max(data, axis=0)

    XX = np.linspace(x_min - 1, x_max + 1)
    YY = -(w[0] / w[1]) * XX - b / w[1]
    plt.plot(XX, YY, '-c', label='Hyperplane')
    plt.show()


# 优化目标： MIN L(w,b) = (-1)*yi * (w * xi + b) 最小
def loss_fun(data, target, w, b):
    loss = 0.0
    for i in range(len(target)):
        loss += (-1) * target[i] * (np.dot(w, data[i].T) + b)

    return loss


# 初始化 w , b
w = np.array((0.0, 0.0))
b = 0.0

# 所有数据训练10轮
num_epochs = 10

# 每10条数据为一个周期
epoch = 20

# 学习率 0.01
lr = 0.01

while (num_epochs <= 30):
    # 随机打乱数据
    data, target = sklearn.utils.shuffle(data, target)
    # 记录已经训练过的数据的条数
    tarin_data_num = 0
    for i in range(len(target)):
        # 分类错误，感知机
        if target[i] * (np.dot(w, data[i].T) + b) <= 0:
            w = w + lr * target[i] * data[i]
            b = b + lr * target[i]
        if ((i + 1) % epoch == 0):
            print("第{}轮训练，after第{}条数据，分类效果如图,w:{},b:{},损失函数值为{}：".format(num_epochs, (i + 1), w, b,
                                                                         loss_fun(data, target, w, b)))
            #show_data_line(data, target, w, b)

    num_epochs += 1
