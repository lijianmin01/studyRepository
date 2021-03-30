import pandas as pd
from tqdm import tqdm  # 进度条模块

from glob import glob
lst_file = glob("/home/lijianmin/temp/temp_dataSet/kaggle1/temp/*.csv")

data = pd.DataFrame()
for in_file in tqdm(lst_file):  # 对迭代器添加进度条显示

    df = pd.read_csv(in_file)
    df_data = pd.DataFrame(df)
    data = pd.concat([data,df_data])
data.to_csv("0329csv.csv",index = False)
