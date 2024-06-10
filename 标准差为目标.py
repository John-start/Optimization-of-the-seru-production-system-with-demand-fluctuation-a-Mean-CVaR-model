"""
Author: TR
Created on: 2023-12-18 18:46:00
Version: 4.0

相比较之前的版本，每次运行都随机生成批量大小，根据每组的数据计算改组的工期，
多次运行，得到每组数据的在该方案下的均值与方差，作为优化目标
采用NSGA-II算法对均值与方差两个目标进行优化
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA 
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
import random
import copy
import seru
import generate

# 设置的数据组数
num = 100
alpha = 0.75

# generate.generate(num)
# 产品各道工序标准工时
time_info = pd.read_csv('赛汝生产系统/算例测试/time_info.csv')
time_info = dict(zip(zip(time_info['产品类型'],time_info['工序']),time_info['标准工时']))
# 员工各技能熟练程度
skill_info = pd.read_csv('赛汝生产系统/算例测试/skill_info.csv')
skill_info = skill_info.drop(columns=['员工'])
skill_info = skill_info.to_numpy()
# 产品切换时间
setup_time = pd.read_csv('赛汝生产系统/算例测试/setup_time.csv')
setup_time = dict(zip(setup_time['类型'],setup_time['时间']))
W,L = skill_info.shape  # 工人数和工序数
N = int(len(time_info)/L)  # 产品类型数
random_info = pd.read_csv('赛汝生产系统/算例测试/random_info.csv')
M = len(random_info)  # 批次数

# np.random.seed(1)  # 固定随机种子，便于复现
def gen_chrom():
    x = np.array([],dtype=int)
    sequence1 = np.random.permutation(list(range(1,2*W)))
    x=sequence1
    return x
# a = gen_chrom()
# print(a)

def get_duration(x,batch_info,batch_list):
    """染色体评价函数"""
    # 染色体解码
    formation = [i if i <= W else 0 for i in x]
    # 使用列表切片根据 0 切割成多个子列表
    formation = [formation[i:j] for i, j in zip([0] + [idx + 1 for idx, val in enumerate(formation) if val == 0],
    [idx for idx, val in enumerate(formation) if val == 0] + [None])]
    # 删除多余空列表
    formation = [sublist for sublist in formation if sublist]
    J = len(formation)  # 赛汝个数
    seru_list = [seru.Seru() for _ in range(J)]  # 赛汝列表
    staff_list = [seru.Staff(i) for i in range(1,W+1)]
    # 赛汝内加入工人
    for i,sub in enumerate(formation):
        for j,subsub in enumerate(sub):
            seru_list[i].add_staff(staff_list[subsub-1])
            
    # 赛汝内加入产品批次
    scheduling = list(range(1,M+1))
    seru_time = [0]*J  # 记录各个赛汝的加工时间
    for i in scheduling:
        assign_seru = np.argmin(seru_time)
        seru_list[assign_seru].add_batch(batch_list[i-1],time_info,skill_info,L,setup_time)
        # 更新工时列表
        for j in range(J):
            seru_time[j] = seru_list[j].process_time

    return max(seru_time)

def calculate_cvar(data, alpha):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    var_index = int(np.floor(alpha * n))
    cvar = np.round(np.mean(sorted_data[var_index:]),2)
    return cvar
def evaluate(x,num):
    ans = []
    for i in range(num):
        file_name = '赛汝生产系统/算例测试/数据/batch_info'+str(i+1)+'.csv'
        batch_info = pd.read_csv(file_name)
        batch_list = [seru.Batch(row['批次'], row['批量大小'], row['类型']) for _, row in batch_info.iterrows()]
        a = get_duration(x,batch_info,batch_list)
        ans.append(a)
    ans = np.array(ans)
    mean = round(ans.mean(),2)
    std = round(ans.std(),2)
    return mean,std

def encode(solution):
    chrom = []
    for i,sub_list in enumerate(solution):
        chrom = chrom + sub_list + [W+1+i]
    chrom = chrom + list(range(len(chrom)+1,2*W))
    return chrom

class Myproblem(ElementwiseProblem):
    
    def __init__(self,W,M,**kwargs):
        self.W = W
        self.M = M
        chrom = [0]*(2*W-1)
        super().__init__(vars = chrom,n_obj=2,xl=1,xu=2*W-1,**kwargs)
    
    def _evaluate(self, x, out, *args, **kwargs):
        f1,f2 = evaluate(x,num)
        out['F'] = [f1,f2]
        

class Mysampling(Sampling):
    
    def _do(self,problem,n_samples,**kwargs):
        X = np.full((n_samples,2*W-1),0,dtype=int)
        for i in range(n_samples):
            X[i,:] = gen_chrom()
        # X[0,:] = encode([[17, 3, 9, 2, 7, 10, 18, 16, 6, 5, 13], [14, 19, 1, 11, 12, 15, 20, 8, 4]])
        # X[1,:] = encode([[17, 14, 9, 10, 2, 7, 18, 16, 6, 5, 13], [3, 19, 1, 11, 12, 15, 20, 8, 4]])
        # X[2,:] = encode([[3, 5, 2, 7, 18, 13, 15, 16, 1, 17, 6], [14, 19, 10, 20, 9, 11, 8, 12, 4]])
        # X[3,:] = encode([[3, 5, 2, 7, 18, 16, 6, 17, 9, 13, 15], [11, 19, 14, 1, 20, 12, 8, 4, 10]])
        # X[4,:] = encode([[17, 9, 2, 7, 18, 16, 6, 1, 5, 13, 19], [11, 8, 14, 10, 20, 15, 12, 3, 4]])
        # X[5,:] = encode([[9, 2, 7, 10, 18, 16, 6, 5, 19, 13], [14, 1, 15, 11, 4, 20, 8, 12, 17, 3]])
        # X[6,:] = encode([[3, 9, 2, 7, 10, 18, 16, 6, 5, 19, 13], [14, 1, 15, 11, 20, 8, 12, 17, 4]])
        # X[7,:] = encode([[17, 3, 9, 2, 7, 10, 18, 16, 6, 5, 13], [14, 19, 1, 11, 12, 15, 20, 8, 4]])
        # X[8,:] = encode([[17, 3, 9, 4, 7, 10, 18, 16, 6, 5, 13], [14, 19, 1, 11, 15, 20, 8, 12, 2]])
        # X[9,:] = encode([[17, 9, 2, 6, 10, 18, 16, 7, 5, 19, 13], [14, 1, 15, 11, 4, 20, 8, 12, 3]])
        # X[10,:] = encode([[3, 5, 2, 7, 9, 17, 18, 16, 4, 19, 13], [6, 11, 14, 1, 20, 12, 8, 10, 15]])
        # X[11,:] = encode([[9, 2, 7, 4, 10, 18, 16, 6, 5, 13], [14, 19, 1, 11, 15, 20, 8, 12, 17, 3]])
        return X

def order_crossover(p1,p2,point_1,point_2):
    """顺序交叉。将父代染色体1该区域内的基因复制到子代1相同位置上，再在父代染色体2上将子代1中缺少的基因按照顺序填入。

    Args:
        p1 (_type_): 父代1
        p2 (_type_): 父代2
        point_1 (_type_): 交叉点1
        point_2 (_type_): 交叉点2
    """
    p1_seg = p1[point_1:point_2]
    p2_seg = p2[point_1:point_2]
    off1_seg,off2_seg = [],[]  # 子代的片段
    
    off1_seg = [i for i in p2 if i in p1_seg]  # 子代1的片段
    off2_seg = [i for i in p1 if i in p2_seg]  # 子代2的片段
    
    off1,off2 = copy.deepcopy(p1),copy.deepcopy(p2)  # 子代个体
    for i in range(point_1,point_2):
        off1[i] = off1_seg[i-point_1]
        off2[i] = off2_seg[i-point_1]
    return off1,off2

class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2,2)
    
    def _do(self,problem,X,**kwargs):
        _,n_matings,n_var = X.shape
        Y = np.full_like(X,0,dtype=int)
        for k in range(n_matings):
            a,b = X[0,k,:],X[1,k,:]
            # 构造部分
            point_1 = random.randint(0,2*W-2)
            point_2 = random.randint(0,2*W-2)
            point_min = min(point_1,point_2)
            point_max = max(point_1,point_2)
            off_a,off_b = copy.deepcopy(order_crossover(a,b,point_min,point_max))
            Y[0, k, :], Y[1, k, :] = off_a, off_b
        return Y

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self,problem,X,**kwargs):
        for i in range(len(X)):
            r = np.random.random()
            if r < 0.05:
                m11 = random.randint(0,2*W-2)
                m12 = random.randint(0,2*W-2)
                m11,m12 = min(m11,m12),max(m11,m12)
                p = X[i,:]
                p[m11],p[m12] = p[m12],p[m11]
                X[i,:] = p
        return X

m = Myproblem(W=W,M=M)
algorithm = NSGA2(
    pop_size=50,
    sampling=Mysampling(),
    crossover=MyCrossover(),
    mutation=MyMutation(), 
    eliminate_duplicates=True)

from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination
termination = get_termination("time", "00:00:150")
res = minimize(
        m,
        algorithm,
        termination=termination,
        verbose=True,
        save_history=False
        )

# plot = Scatter()
# plot.add(m.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.show()

def decode(x):
    # 染色体解码
    formation = [i if i <= W else 0 for i in x] 
    # 使用列表切片根据 0 切割成多个子列表
    formation = [formation[i:j] for i, j in zip([0] + [idx + 1 for idx, val in enumerate(formation) if val == 0],
    [idx for idx, val in enumerate(formation) if val == 0] + [None])]
    # 删除多余空列表
    formation = [sublist for sublist in formation if sublist]
    return formation

schemes = []  # 存储所有染色体解码后的方案

# print('解码结果为：')
for i in range(len(res.X)):
    schemes.append(decode(res.X[i]))

# print(res.X)

def remove_equivalent_schemes(schemes):
    """这里考虑赛汝的顺序"""
    unique_schemes = set()
    for scheme in schemes:
        unique_scheme = frozenset(tuple(sorted(box)) for box in scheme)
        unique_schemes.add(tuple(sorted(unique_scheme)))
    return [list(list(box)) for box in unique_schemes]

result = remove_equivalent_schemes(schemes)
print('去重后的方案为:')
print(result)

duration_mean,duration_std = set(),set()
for i in range(res.F.shape[0]):
    duration_mean.add(res.F[i][0])
    duration_std.add(res.F[i][1])
print('最大完工时间的均值与方差为:')
print(duration_mean,duration_std)



def evaluate(x,num):
    ans = []
    for i in range(num):
        file_name = '赛汝生产系统/算例测试/数据/batch_info'+str(i+1)+'.csv'
        batch_info = pd.read_csv(file_name)
        batch_list = [seru.Batch(row['批次'], row['批量大小'], row['类型']) for _, row in batch_info.iterrows()]
        a = get_duration(x,batch_info,batch_list)
        ans.append(a)
    ans = np.array(ans)
    mean = round(ans.mean(),2)
    std = round(ans.std(),2)
    cvar = calculate_cvar(ans,alpha)
    return mean,std,cvar

ans1111 = []
print('-------采用标准差为目标求解出的%f个帕累托前沿解的各项信息----------'%len(res.X))
for i in range(len(res.X)):
    ans1111.append(evaluate(res.X[i],num))
    print('均值为%f,标准差为%f,cvar为%f'%(evaluate(res.X[i],num)))