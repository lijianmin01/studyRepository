for user in users:
        print("用户{}推荐的商品：".format(user))
        for way in ways:
            print("   方式{}：".format(way), end=" ")
            ub = ItemCF(user_score_dict=user_score_dict,way=way)
            print(ub.recommend(user))