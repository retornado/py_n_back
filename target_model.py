# -*- coding: utf-8 -*-
'''
裂变靶飞行时间方法测量建模
对象：
1. 靶片
    中子裂变截面 b(En)
    活性区面积   Aeff
    U235面密度   pd
    裂变靶原子量 A
    
2. PIN探测器
    平均沉积能量 Edep
    灵敏面半径   r
    电子-空穴对生成平均能  w

3. 几何参数
    靶片中心和PIN探测器灵敏面间距  L

'''
import tof
from scipy import constants as scipy_cons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################################################################################
# 物理常数
neutron_m = scipy_cons.physical_constants['neutron mass'][0]
light_c = scipy_cons.physical_constants['natural unit of velocity'][0]
unit_e =  scipy_cons.physical_constants['elementary charge'][0]
NA = scipy_cons.physical_constants['Avogadro constant'][0]

###################################################################################
# 载入截面数据库
def cs_section(filename):
    junk = pd.read_table(filename,skipinitialspace=True,sep = ' ')  
    junk = junk.drop(0).reset_index()
    junk['en'] = junk['Neutron']
    junk['cs'] = junk['Energy']
    data = junk[['en','cs']].copy()
    data['en'] = data['en'].map(lambda x : eval(x))
    data['cs'] = data['cs'].map(lambda x : eval(x))
    return data

# U235中子裂变截面
filename_u235 = r'D:\92-U-235_18.txt'
cs_data_u235 = cs_section(filename_u235)

# CH2反冲质子截面
filename_ch2 =  r'D:\01-H-1_2.txt'
cs_data_ch2 = cs_section(filename_ch2)

# en  -- 选出en中子的截面
def select_cs_from_en(en,data):
    junk = data.copy()
    junk['en_diff'] = np.abs(junk['en'] - en*10**6)
    return junk[junk['en_diff'] == min(junk['en_diff'])]
    
###################################################################################
# 探测器 
##. 裂变靶 
class U_slab():
    # input : pd : mg/cm2  , phi : mm
    def __init__(self,pd = 0.58,phi =30,A=235.0):
        self.pd = pd/1000.0  # g/cm2
        self.Aeff = np.pi*(phi/20.0)**2   #cm2
        self.A = A
        # 裂变能谱高到110MeV（Ufrag计算），低到20MeV（标定实验）
        self.mean_Edep = 70 
        self.e_dep_high = 110
        self.e_dep_low = 20 
        self.cs_data = cs_data_u235
    
    def cs_database(self,en_low = 0.2,en_high = 16):
        # 选择特定中子能量区间的散射截面
        self.cs = self.cs_data[self.cs_data['en'] >= en_low*10**6].copy()
        self.cs = self.cs[self.cs['en'] <= en_high*10**6].copy()
        self.cs['en'] = self.cs['en']/10**6         

## 反冲质子靶
class CH2_slab():
    def __init__(self,pd = 2,phi =10,A=14.0,theta_slab = 90,theta_pin = 45):
        # phi mm
        self.pd = pd/1000.0  # g/cm2
        theta_rad =  np.deg2rad(theta_slab)
        self.Aeff_real = np.pi*(phi/20.0)*(phi*np.sin(theta_rad)/20.0)  #cm2
        self.Aeff = np.pi*(phi/20.0)**2
        self.A = A
        self.theta = theta_pin
        # 反冲质子能谱高到7MeV,和PIN厚度关系最大，需要模拟计算
        self.mean_Edep = 3  
        self.e_dep_high = 7
        self.e_dep_low = 0.5        
        self.cs_data = cs_data_ch2        
        
    def cs_en_theta(self,en):
        # 对应公式2,质子反冲截面，和中子能量，出射角度有关
        cs = select_cs_from_en(en,self.cs_data)['cs'].values[0]
        theta_rad =  np.deg2rad(self.theta)
        junk = (en/90.0)**2*2.0
        junk1 = (1+junk*np.cos(2*theta_rad)**2)/(1+junk/3.0)
        # 本来末尾除以pi，结果的单位是4pi；这里为了和U一样，变成乘4，结果单位是1
        return cs*np.cos(theta_rad)*junk1*4
    
    def cs_database(self,en_low = 0.2,en_high = 16):
        # cross section database : en > en_low (MeV)
        self.cs = self.cs_data[self.cs_data['en'] >= en_low*10**6].copy()
        self.cs = self.cs[self.cs['en'] <= en_high*10**6].copy()
        self.cs['en'] = self.cs['en']/10**6
        self.cs['cs_theta'] = self.cs['en'].map(lambda x : self.cs_en_theta(x))
        # 方便统一的Detector_Ana类计算
        self.cs['cs'] =  self.cs['cs_theta']

## PIN
class PIN():
    # input : phi : mm  , Edep
    def __init__(self,phi =30,L = 150):
        self.r = phi/2.0   # mm
        self.L = L   #mm
        self.w = 3.63  # eV   在 PIN中产生一对电子-空穴对需要的平均能量     
        self.fmwh = 16   # ns    # PIN输出信号宽度16
        unit_value = ['MeV','bar(10e-24 cm2)','/cm2/ns','ns','cm2','/ns','ns','hour',
                    '%','nA']
        unit_items = ['en','cs','flux_time','flight_time','detec_eff','detec_rate',
                    'delta_t','count_time','over_lap_rate','current']
        # 各个量的单位
        self.unit_se = pd.Series(unit_value,index = unit_items)
###################################################################################
# 分析 
class Detector_Ana():
    def __init__(self,slab,pin,flux_numb = 4,counts_all = 4500,en_section = [0.2,16]):
        self.slab = slab
        self.slab.cs_database(en_low = en_section[0],en_high = en_section[1])
        self.cs = self.slab.cs
        self.pin = pin
        self.flux_numb = flux_numb
        self.counts_all = counts_all
        self.flight_length = tof.CSNS_source.flux_option.loc[self.flux_numb]['distance']
        self.analyze()

    def analyze(self):
        # 每ns中子个数，能量为en ，单位 ：/cm2/ns
        # 能量为en中子，飞到测点处的通量
        self.cs['flux_time'] = self.cs['en'].map(lambda x : tof.select_flux_from_en(x,self.flux_numb)['flux_time'])
        # 能量为en的中子，到达探测器的飞行时间 ns
        self.cs['flight_time'] = self.cs['en'].map(lambda x : tof.relative_t(x,flight_length = self.flight_length))        
        # PIN探测器的探测效率（不考虑源强）
        self.cs['detec_eff'] = self.slab.pd*self.slab.Aeff*NA*self.cs['cs']*10**(-24)*self.pin.r**2/self.slab.A/self.pin.L**2/2.0
        # 考虑源强后，在PIN探测器上的单粒子脉冲发生速率 个/ns
        self.cs['detec_rate'] = self.cs['flux_time']*self.cs['detec_eff']  
        # 根据脉冲宽度（前放主放引起）和脉冲发生速率估计重叠率 %
        self.cs['over_lap_rate'] = (1 - np.exp(-1*self.cs['detec_rate']*self.pin.fmwh))*100        
        # 估计输出电流
        self.cs['current'] = 10**9*self.cs['detec_rate']*self.slab.mean_Edep*10**6*unit_e/self.pin.w*10**9   # C/s
    
    
    def constant_time(self,time = 300):
        # 给定时间，单能中子能够达到的能量分辨
        self.count_time = time
        self.cs['delta_t_constant_time'] = self.counts_all/(time*60*60*25*1.0)/self.cs['detec_rate']
        self.cs['resol_e_constant_time'] = 2*self.cs['delta_t_constant_time']/self.cs['flight_time']*100
    
    def volt(self,resis = 50):
        # 裂变碎片最大沉积能量对应电量，单位为C
        self.q_high = self.slab.e_dep_high*10**6*unit_e/self.pin.w
        self.q_low =  self.slab.e_dep_low*10**6*unit_e/self.pin.w
        # 根据脉冲宽度估计最大幅度, mV
        self.v_high = 10**12*self.q_high*resis/self.pin.fmwh
        self.v_low = 10**12*self.q_low*resis/self.pin.fmwh
        # 根据前放参数估计最大幅度, mV 
        self.v_high1 = self.q_high*12*10**15
        # 数采量程800mV，除以电量，可估计前放 + 主放的放大倍数
        # 即电荷灵敏度，单位“mV/fC, 数采仪的幅度为800mV，根据单粒子脉冲最大电量估计
        self.charge_sensi = 800.0/(self.q_high*10**15)
            
    def plot(self,x = 'en',y = 'resol_e_constant_time',x_add = '',y_add=''):
        plt.plot(self.cs[x],self.cs[y],marker = 'o',linestyle = '-')
        plt.xlabel(x+x_add)
        plt.ylabel(y+y_add)
        
    