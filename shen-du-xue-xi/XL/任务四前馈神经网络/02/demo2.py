import pickle
# pickle.dump(data,open('file_path','wb'))  #后缀.pkl可加可不加
# 云服务器版本

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time
# import matplotlib.pyplot as plt


# 获取手写数字的数据集
def get_mnist_data():
    # 每一组有多少数据
    batch_size = 100

    # 获取数据集
    train_dataset = torchvision.datasets.MNIST(root='/root/temp_dataSet/', train=True,
                                               transform=transforms.ToTensor(),
                                               download=False)
    test_dataset = torchvision.datasets.MNIST(root='/root/temp_dataSet/', train=True,
                                              transform=transforms.ToTensor(),
                                              download=False)
    # 加载数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# 获取手写数字的数据集
def get_mnist_data_1():
    # 每一组有多少数据
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 获取数据集
    train_dataset = torchvision.datasets.MNIST(root='/root/temp_dataSet/', train=True,
                                               transform=transform,
                                               download=False)
    test_dataset = torchvision.datasets.MNIST(root='/root/temp_dataSet/', train=True,
                                              transform=transform,
                                              download=False)
    # 加载数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader




# 全链接神经网络  784  512  256  128  64  10
# ReLu
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 全链接神经网络  784  512  256  128  64  10
# ReLu
class Net_relu(nn.Module):
    def __init__(self):
        super(Net_relu, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 全链接神经网络  784  512  256  128  64  10
# sigmoid
class Net_sigmoid(nn.Module):
    def __init__(self):
        super(Net_sigmoid, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 全链接神经网络  784  512  256  128  64  10
# Tanh
class Net_Tanh(nn.Module):
    def __init__(self):
        super(Net_Tanh, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 全链接神经网络  784  512  256  128  64  10
# Softplus
class Net_Softplus(nn.Module):
    def __init__(self):
        super(Net_Softplus, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 全链接神经网络  784  512  256  128  64  10
# Softplus
class Net_Softplus(nn.Module):
    def __init__(self):
        super(Net_Softplus, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 激活函数
def jihuo_train_model(model_type):
    model = None
    if model_type == 1:
        model = Net_relu()
    elif model_type == 2:
        model = Net_sigmoid()
    elif model_type == 3:
        model = Net_Tanh()
    elif model_type == 4:
        model = Net_Softplus()

    learning_rate = 0.001


    # 定义损失函数和优化算法
    ## 学习率

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # 训练数据的大小，也就是含有多少个barch
    total_step = len(train_loader)

    # 所有的数据一共训练几轮 5
    num_epochs = 4

    # 用一个数组计算每经过 500条数据训练过后，模型自爱测试集上的正确性
    accuracy_list = []

    for epoch in range(num_epochs):
        # 第几个barch 和 （图像、标签）
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # 前向传播
            prv_labels = model(imgs)
            # 计算损失
            loss = criterion(prv_labels, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{} / {}],Srep [{} / {}],loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, total_step,
                                                                          loss.item()))

            if (i + 1) % 5 == 0:
                # 测试模型，不用计算梯度
                with torch.no_grad():
                    # 被预测争取的个数
                    correct = 0
                    # 预测的图像总数
                    total = 0
                    for imgs, labels in test_loader:
                        imgs = imgs.reshape(-1, 28 * 28)
                        labels = labels

                        prvs = model(imgs)
                        _, predicted = torch.max(prvs.data, 1)
                        # 测速图片的数目
                        total += labels.size(0)
                        # 正确分类的图片数目
                        correct += (predicted == labels).sum().item()

                    # 正确率
                    acc = 1.0 * correct / total
                    accuracy_list.append(acc)
                    # print("正确率为：{}".format(acc))
    import pickle
    f = open('jihuo_{}.pkl'.format(model_type), 'wb')
    pickle.dump(accuracy_list, f)
    f.close()
    return accuracy_list


# lr
def lr_train_model(model_type):
    model = Net1()
    if model_type == 1:
        learning_rate = 0.001
    elif model_type == 2:
        learning_rate = 0.01
    elif model_type == 3:
        learning_rate = 0.1


    # 定义损失函数和优化算法
    ## 学习率

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # 训练数据的大小，也就是含有多少个barch
    total_step = len(train_loader)

    # 所有的数据一共训练几轮 5
    num_epochs = 4

    # 用一个数组计算每经过 500条数据训练过后，模型自爱测试集上的正确性
    accuracy_list = []

    for epoch in range(num_epochs):
        # 第几个barch 和 （图像、标签）
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # 前向传播
            prv_labels = model(imgs)
            # 计算损失
            loss = criterion(prv_labels, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{} / {}],Srep [{} / {}],loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, total_step,
                                                                          loss.item()))

            if (i + 1) % 5 == 0:
                # 测试模型，不用计算梯度
                with torch.no_grad():
                    # 被预测争取的个数
                    correct = 0
                    # 预测的图像总数
                    total = 0
                    for imgs, labels in test_loader:
                        imgs = imgs.reshape(-1, 28 * 28)
                        labels = labels

                        prvs = model(imgs)
                        _, predicted = torch.max(prvs.data, 1)
                        # 测速图片的数目
                        total += labels.size(0)
                        # 正确分类的图片数目
                        correct += (predicted == labels).sum().item()

                    # 正确率
                    acc = 1.0 * correct / total
                    accuracy_list.append(acc)
                    # print("正确率为：{}".format(acc))
    import pickle
    f = open('lr_{}.pkl'.format(model_type), 'wb')
    pickle.dump(accuracy_list, f)
    f.close()
    return accuracy_list


# 数据是否预处理
def deal_train_model(train_loader,test_loader,model_type):
    model = Net1()
    learning_rate = 0.001

    # 定义损失函数和优化算法
    ## 学习率

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # 训练数据的大小，也就是含有多少个barch
    total_step = len(train_loader)

    # 所有的数据一共训练几轮 5
    num_epochs = 4

    # 用一个数组计算每经过 500条数据训练过后，模型自爱测试集上的正确性
    accuracy_list = []

    for epoch in range(num_epochs):
        # 第几个barch 和 （图像、标签）
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # 前向传播
            prv_labels = model(imgs)
            # 计算损失
            loss = criterion(prv_labels, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{} / {}],Srep [{} / {}],loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, total_step,
                                                                          loss.item()))

            if (i + 1) % 5 == 0:
                # 测试模型，不用计算梯度
                with torch.no_grad():
                    # 被预测争取的个数
                    correct = 0
                    # 预测的图像总数
                    total = 0
                    for imgs, labels in test_loader:
                        imgs = imgs.reshape(-1, 28 * 28)
                        labels = labels

                        prvs = model(imgs)
                        _, predicted = torch.max(prvs.data, 1)
                        # 测速图片的数目
                        total += labels.size(0)
                        # 正确分类的图片数目
                        correct += (predicted == labels).sum().item()

                    # 正确率
                    acc = 1.0 * correct / total
                    accuracy_list.append(acc)
                    # print("正确率为：{}".format(acc))
    import pickle
    f = open('deal_{}.pkl'.format(model_type), 'wb')
    pickle.dump(accuracy_list, f)
    f.close()
    return accuracy_list


if __name__ == '__main__':
    print("start~")
    # 如果可以调用gpu的话，使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取训练数据和测试数据
    train_loader, test_loader = get_mnist_data()

    '''不同的学习率 0.001  0.01  0.1'''
    lr_acc_list = []
    time.sleep(10)
    for net_i in which_net:
        cnt = lr_train_model(net_i)
        lr_acc_list.append(cnt)
        time.sleep(10)

    f = open('lr_acc.pkl', 'wb')
    pickle.dump(lr_acc_list, f)
    f.close()
    # # 画图
    # draw_acc(acc_list)
    print("end~")

    '''不同的激活函数'''
    which_net = [i for i in range(1, 5)]

    jihuo_acc_list = []
    time.sleep(10)
    for net_i in which_net:
        cnt = jihuo_train_model(net_i)
        jihuo_acc_list.append(cnt)
        time.sleep(10)

    f = open('jihuo_acc.pkl', 'wb')
    pickle.dump(jihuo_acc_list, f)
    f.close()

    '''数据是否进行去处理'''
    deal_acc_list = []
    time.sleep(10)
    cnt1 = deal_train_model(train_loader, test_loader,1)
    deal_acc_list.append(cnt1)
    time.sleep(10)

    train_loader, test_loader = get_mnist_data_1()
    # 预处理的数据
    time.sleep(10)
    cnt2 = deal_train_model(train_loader, test_loader, 1)
    deal_acc_list.append(cnt2)
    time.sleep(10)
    f = open('deal_acc.pkl', 'wb')
    pickle.dump(deal_acc_list, f)
    f.close()