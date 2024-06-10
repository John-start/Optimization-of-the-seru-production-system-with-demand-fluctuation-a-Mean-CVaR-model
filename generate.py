import numpy as np
import pandas as pd

def generate(num):
    """生成num组数据

    Args:
        num (int): 数据组数
    """
    # np.random.seed(42)  # 固定随机数种子
    for i in range(num):
        random_info = pd.read_csv('./赛汝生产系统/算例测试/random_info.csv')
        random_values = np.random.normal(loc=random_info['批量大小均值'], scale=random_info['批量大小标准差'])
        random_info['批量大小'] = np.round(random_values).astype(int)
        random_info = random_info.drop(columns=['批量大小均值','批量大小标准差'])
        file_name = './赛汝生产系统/算例测试/数据/batch_info'+str(i+1)+'.csv'
        random_info.to_csv(file_name,index=False)
generate(100)