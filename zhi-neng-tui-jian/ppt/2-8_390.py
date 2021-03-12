# -*- coding: utf-8 -*-
"""
    Desc:
        代码2-8   PersonalRank算法原理
"""
import time


def PersonalRank(G, alpha, root, max_step):
    rank = dict()
    rank = {x: 0 for x in G.keys()}
    rank[root] = 1
    # 开始迭代
    begin = time.time()
    for k in range(max_step):
        tmp = {x: 0 for x in G.keys()}
        # 取节点i和它的出边尾节点集合ri
        for i, ri in G.items():
            # 取节点i的出边的尾节点j以及边E(i,j)的权重wij, 边的权重都为1，归一化之后就是1/len(ri)
            for j, wij in ri.items():
                # i是j的其中一条入边的首节点，因此需要遍历图找到j的入边的首节点，
                # 这个遍历过程就是此处的2层for循环，一次遍历就是一次游走
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
        # 我们每次游走都是从root节点出发，因此root节点的权重需要加上(1 - alpha)
        tmp[root] += (1 - alpha)
        rank = tmp
    end = time.time()
    print('use time', end - begin)
    rank = sorted(rank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return rank


if __name__ == '__main__':
    alpha = 0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}

    # rank = PersonalRank(G, alpha, 'b', 50)  # 从'b'节点开始游走
    # print(rank)

    items_dict = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    rank = PersonalRank(G, alpha, 'A', 50) # 从'A'节点开始游走
    print(rank)
    # for k in items_dict.keys():
    #     if k in rank:
    #         items_dict[k] = rank[k]
    # # sort:
    # result = sorted(items_dict.items(), key=lambda d: d[1], reverse=True)
    # print("\nThe result:")
    # for k in result:
    #     print("%s:%.3f \t" % (k[0], k[1]))
