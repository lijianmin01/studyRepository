import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 画图
def draw_acc(acc):
    net_num_change = ['784->1024->10', '784->512->256->128->64->10', '487->32->32->32->32->10', '784->1024->128->10','784->2048->10']

    plt.figure()

    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(2)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数

    plt.ylim(0,110)

    plt.ylabel('Squares', fontsize=1)


    x = [i for i in range(len(acc[0]))]

    for net_type in range(len(acc)):
        y = [float(i)*100 for i in acc[net_type]]
        plt.plot(x, y,linewidth=1)
        # plt.scatter(x, y)

    plt.xlabel('batch(500)')
    plt.ylabel('Accuracy (%)')

    # 图示：在左上角显示图示神经网络名字
    class_labels = ["net{}:{}".format(i, net_num_change[i]) for i in range(len(net_num_change))]
    plt.legend(class_labels)

    plt.savefig('plot.pdf')
    plt.show()

if __name__ == '__main__':
    f = open('../03/width_and_deep/acc.pkl', 'rb')
    a = pickle.load(f)
    f.close()

    draw_acc(a)