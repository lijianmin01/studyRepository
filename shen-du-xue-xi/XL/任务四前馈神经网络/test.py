# import matplotlib.pyplot as plt
#
# def draw_acc(acc):
#     plt.figure()
#
#     x = [i for i in range(len(acc[0]))]
#
#     for net_type in range(len(acc)):
#         y = acc[net_type]
#         plt.plot(x,y)
#         plt.scatter(x,y)
#
#
#     plt.xlabel('batch(500)')
#     plt.ylabel('Accuracy')
#
#     # 图示：在左上角显示图示神经网络名字
#     class_labels = ["net{}".format(i) for i in x]
#     plt.legend(class_labels, loc=2)
#
#     plt.show()

# import pickle
#
# a = pickle.load(open('accuracy_list1.pkl','rb'))
#
# print(a)
#
#
# if __name__ == '__main__':
#     acc = [[1,2,3,4],[10,20,30,40],[0,10,100,100],[6,-10,20,60]]
#     draw_acc(acc)


import pickle
f = open('accuracy_list102.pkl', 'rb')
a = pickle.load(f)
f.close()

print(a)
