class Staff():
    
    def __init__(self,id):
        self.id = id
        # self.skill = skill

class Batch():
    
    def __init__(self,id,mean,type):
        self.id = id
        self.batch_size_mean = mean
        # self.batch_size_std = std
        self.type = type

class Seru():
    def __init__(self) -> None:
        self.staff_list = []  # 员工列表
        self.batch_list = []  # 批次列表
        self.type_list = []  # 产品类型列表
        self.process_time = 0  # 加工时间列表
    def add_staff(self,staff):
        self.staff_list.append(staff.id)
    
    def add_batch(self,batch,time_info,skill_info,L,setup_time):
        self.batch_list.append(batch.id)
        self.type_list.append(batch.type)
        # 总加工时长加上批次时长
        self.process_time += self.get_TF(batch,time_info,skill_info,L)+self.get_TS(batch,setup_time)
        
    def display(self):
        # 打印该赛汝相关信息 
        print('=====================某个赛汝基本信息如下：====================')
        print("员工列表：",self.staff_list)
        print("批次列表：",self.batch_list)
        print("产品类型列表：",self.type_list)
        print("加工时长列表：",self.process_time)
    
    def get_TT(self,batch,time_info,skill_info,L):
        TT_down,TT_up =0,0
        TT_down = len(self.staff_list)  # 工人数量
        for i in range(TT_down):
            for j in range(L):
                TT_up += time_info[(batch.type,j+1)]*skill_info[self.staff_list[i]-1,j]
        return TT_up/TT_down
    
    def get_TF(self,batch,time_info,skill_info,L):
        TF_up,TF_down = 0,0
        TF_up = batch.batch_size_mean*self.get_TT(batch,time_info,skill_info,L)
        TF_down = len(self.staff_list)
        return TF_up/TF_down
    
    def get_TS(self,batch,setup_time):
        # 计算批次切换时间
        if self.batch_list[0] == batch.id:
            return 0
        elif self.type_list[self.batch_list.index(batch.id)-1] == batch.type:
            return 0
        else:
            return setup_time[batch.type]
                
    def get_TB(self,batch_list,batch,time_info,skill_info,L,setup_time):
        TB = 0
        if self.batch_list[0] == batch.id:
            return 0
        else:
            for i in range(self.batch_list.index(batch.id)):
                TB += self.get_TF(batch_list[i],time_info,skill_info,L) + self.get_TS(batch_list[i],setup_time)
            return TB

