import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 从pkl文件中提取值
def get_pkl(file_path):
    f = open(file_path,'rb')
    data = pickle.load(f)
    f.close()

    return data

# 画图
def draw_acc(acc,net_num_change,name):

    plt.figure()
    # 把y轴刻度间隔设置为5,并保存在变量中
    y_major_locator = MultipleLocator(5)
    # ax 为两条坐标轴的实例
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    # 设置y轴上的刻度值范围
    plt.ylim(0,110)
    # 设置y轴上的字体和大小
    plt.ylabel('Squares', fontsize=1)

    x = [i for i in range(len(acc[0]))]

    for net_type in range(len(acc)):
        y = [float(i) * 100 for i in acc[net_type]]
        plt.plot(x, y, linewidth=1)

    plt.xlabel('batch(500)')
    plt.ylabel('Accuracy (%)')

    # 图示：在左上角显示图示神经网络名字
    class_labels = ["net{}:{}".format(i, net_num_change[i]) for i in range(len(net_num_change))]
    plt.legend(class_labels)

    plt.savefig('{}.pdf'.format(name))
    plt.show()

if __name__ == '__main__':
    # 模型不同的深度、宽度
    net_num_change = ['784->1024->10', '784->512->256->128->64->10', '784->32->32->32->32->10', '784->1024->128->10','784->2048->10']

    wd_acc = get_pkl('width_and_deep/acc.pkl')

    draw_acc(wd_acc,net_num_change,'模型不同的深度和宽度')

    # 画不同的激活函数
    net_jihuo_func = ['relu','sigmoid','tanh','softplus']

    jihuo_acc = get_pkl('jihuo/jihuo_acc.pkl')

    draw_acc(jihuo_acc,net_jihuo_func,"不同的激活函数")

    # 数据是否进行预处理
    net_pretreatment = ['No preprocessing','Preprocessing']
    pre_acc = get_pkl('deal/deal_acc.pkl')

    draw_acc(pre_acc,net_pretreatment,"是否进行预处理")

    # 不同的学习率
    net_lr = ['0.001','0.01','0.1']
    lr_acc = get_pkl('lr/lr_acc.pkl')
    draw_acc(lr_acc,net_lr,"不同的学习率")


