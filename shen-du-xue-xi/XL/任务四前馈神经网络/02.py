import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pickle

# 获取手写数字的数据集
def get_mnist_data():
    # 每一组有多少数据
    batch_size = 100

    # 获取数据集
    train_dataset = torchvision.datasets.MNIST(root='/home/lijianmin/DataSets/',train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=False)
    test_dataset = torchvision.datasets.MNIST(root='/home/lijianmin/DataSets/',train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=False)
    # 加载数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
    
    return train_loader,test_loader


# 全链接神经网络  784  1024  10
class Net1(nn.Module):

    def __init__(self):
        super(Net1,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,10),
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 全链接神经网络  784  512  256  128  64  10  
class Net2(nn.Module):

    def __init__(self):
        super(Net2,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,10),
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 全链接神经网络  784  32  32  32  32  10  
class Net3(nn.Module):

    def __init__(self):
        super(Net3,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,10),
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 全链接神经网络  784  1024 128 10  
class Net4(nn.Module):

    def __init__(self):
        super(Net4,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10),
        )

    def forward(self,x):
        x = self.model(x)
        return x


def train_model(model_type):
    model = None 
    if model_type==1:
        model = Net1()
    elif model_type==2:
        model = Net2()
    elif model_type==3:
        model = Net3()
    elif model_type==4:
        model = Net4()

    # 定义损失函数和优化算法
    ## 学习率
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # 训练数据的大小，也就是含有多少个barch
    total_step = len(train_loader)

    # 所有的数据一共训练几轮 5
    num_epochs = 5

    # 用一个数组计算每经过 500条数据训练过后，模型自爱测试集上的正确性
    accuracy_list = []

    for epoch in range(num_epochs):
        # 第几个barch 和 （图像、标签）
        for i,(imgs,labels) in enumerate(train_loader):
            imgs = imgs.reshape(-1,28*28).to(device)
            labels = labels.to(device)

            # 前向传播
            prv_labels = model(imgs)
            # 计算损失
            loss = criterion(prv_labels,labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print("Epoch [{} / {}],Srep [{} / {}],loss:{:.4f}".format(epoch+1,num_epochs,i+1,total_step,loss.item()))

            if (i+1) % 200 == 0:
                # 测试模型，不用计算梯度
                with torch.no_grad():   
                    # 被预测争取的个数
                    correct = 0
                    # 预测的图像总数
                    total = 0
                    for imgs,labels in test_loader:
                        imgs = imgs.reshape(-1,28*28)
                        labels = labels

                        prvs = model(imgs)
                        _,predicted = torch.max(prvs.data,1)
                        # 测速图片的数目
                        total += labels.size(0)
                        # 正确分类的图片数目
                        correct += (predicted == labels).sum().item()
                    
                    # 正确率
                    acc = 1.0 * correct / total
                    accuracy_list.append(acc)
                    # print("正确率为：{}".format(acc))
    # pickle.dump(acc_list, open('accuracy_list{}.pkl'.format(model_type), 'wb'))
    import pickle
    f = open('accuracy_list{}.pkl'.format(model_type), 'wb')
    pickle.dump(accuracy_list, f)
    f.close()
    return accuracy_list

# 画图
def draw_acc(acc):
    net_num_change = ['784->1024->10','784->512->256->128->64->10','487->32->32->32->32->10','784->1024->128->10']

    plt.figure()

    x = [i for i in range(len(acc[0]))]

    for net_type in range(len(acc)):
        y = acc[net_type]
        plt.plot(x,y)
        plt.scatter(x,y)

    
    plt.xlabel('batch(500)')
    plt.ylabel('Accuracy')

    # 图示：在左上角显示图示神经网络名字
    class_labels = ["net{}:{}".format(i,net_num_change[i]) for i in range(len(net_num_change))]
    plt.legend(class_labels)
    
    plt.show()

if __name__ == '__main__':

    # 如果可以调用gpu的话，使用gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取训练数据和测试数据
    train_loader,test_loader = get_mnist_data()

    which_net = [i for i in range(1,5)]

    acc_list = []
    
    for net_i in which_net:
        cnt = train_model(net_i)
        acc_list.append(cnt)

    print(acc_list)
    # 画图
    draw_acc(acc_list)


