import pandas as pd
from tqdm import tqdm  # 进度条模块

from glob import glob
lst_file = glob("/zhi-neng-tui-jian/kaggle/code/物品协同过滤算法/temp/*.csv")

data = pd.DataFrame()
for in_file in tqdm(lst_file):  # 对迭代器添加进度条显示

    df = pd.read_csv(in_file)
    df_data = pd.DataFrame(df)
    data = pd.concat([data,df_data])
data.to_csv("test.csv",index = False)
