# 程序实现Swish激活函数，展示不同β参数下的函数图像

# B_list 含有不同的β参数 参数列表

def draw_swish_image(B_list):
    # 引入所用到的包
    import numpy as np
    import matplotlib.pyplot as plt

    # 初始化图像
    plt.figure()
    # 更改坐标轴的位置

    # get current axis 获得坐标轴对象
    ax = plt.gca()

    # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # 定义x 的取值范围
    XX = np.linspace(-6, 6)
    for B in B_list:
        # Swish激活函数 表达式
        YY = XX * (1 / (1 + np.exp(-1.0 * B * XX)))
        plt.plot(XX, YY)

    # 图示：在左上角显示图示B值
    class_labels = ["β="+str(B) for B in B_list]
    plt.legend(class_labels, loc=2)

    plt.show()


if __name__ == '__main__':
    # B_list = [0.1,0.5,1.0,10.0,100.0]
    B_list = [0.1, 1.0, 10.0]
    draw_swish_image(B_list)