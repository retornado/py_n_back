# -*- coding: utf-8 -*-
"""
飞行时间方法建模
@author: luowei

"""
import math
import numpy as np
import pandas as pd
from scipy import constants as scipy_cons
import matplotlib.pyplot as plt

###################################################################################
# 物理常数
neutron_m = scipy_cons.physical_constants['neutron mass'][0]
light_c = scipy_cons.physical_constants['natural unit of velocity'][0]
unit_e =  scipy_cons.physical_constants['elementary charge']

###################################################################################
# 飞行时间测量方法
def flight_t(en,flight_length = 76.6):
    # 单位 ： en / MeV  flight_length/m  t / ns
    return 72.3*flight_length/np.sqrt(en)

def relative_t(en,flight_length = 76.6):
    # en -- > t
    junk = 1-(939.552/(en+939.552))**2
    return flight_length/light_c/np.sqrt(junk)*10**9

def t_to_e(t,flight_length = 76.6):
    # t -- > en(MeV)
    return (72.3*flight_length/t)**2

def relative_resol_e(resol_t,en,flight_length=76.6):
    # 不考虑飞行距离的不确定度
    junk1 = (en+939.552)/en
    tt = relative_t(en,flight_length=flight_length) # ns
    beta = flight_length/light_c/tt*10**9
    junk2 = beta**2/(1-beta**2)
    return resol_t*junk1*junk2
    
def relative_resol_t(resol_e,en,flight_length=76.6):
    # 不考虑飞行距离的不确定度,
    # 由分析知扩大delta_T统计区间，对应的ET平面覆盖面积可以通过扩大delta_E来完成。
    # 故可以反用该关系，通过指定delta_E来推算统计的delta_T区间
    junk1 = (en+939.552)/en
    tt = relative_t(en,flight_length=flight_length) # ns
    beta = flight_length/light_c/tt*10**9
    junk2 = beta**2/(1-beta**2)
    return resol_e/junk1/junk2
    
def nonrelative_resol_e(resol_t):
    return resol_t*2

def distance_diff(en,time_diff):
    junk = 1-(939.552/(en+939.552))**2
    return  light_c*np.sqrt(junk)*time_diff/10**9

###################################################################################
# 白光中子源描述
class Neutron_source():
    def __init__(self,power=25,flux_file = r'D:\en.txt'):
        # 输入: 
        #     加速器功率/每发质子打靶数/每发中子总注量:   
                # 1. 给出这三者转换关系
        #     ES和束斑尺寸选择   
        #  参数存储  
                # 1. 不同ES和束斑尺寸之间的相对比例，尺寸距离信息等
                # 2. 不同ES的归一化中子能谱（flux_area的和为1）
                # 3. 加速器频率
        #  输出：
        #       当前束流参数下的中子能谱 
        self.Freq = 25
        self.filename = flux_file
        self.power = power
        self.read_flux_option()
        self.read_raw_da()
            
    def set_power(self,power):
        self.power = power
        self.read_flux_option()
        self.read_raw_da()
        
        
    def read_flux_option(self):
        # CSNS白光源的各种构型信息
        df = pd.DataFrame(columns = ['spot','spot_area','distance','flux'])
        spot = [u'$\phi$ 20mm',u'$\phi$ 30mm',u'$\phi$ 60mm',u'90*90mm']
        diameter = [1,1.5,3]
        area = [x**2*math.pi for x in diameter]+[81]
        distance = [56,77.6]
        flux = [6.3e+4,2.3e+4,1.1e+6,3.9e+5,2.2e+7,6.8e+6,3.0e+7,1.1e+7]
        for i in range(len(spot)):
            for j in range(len(distance)):
                data = [spot[i],area[i],distance[j],flux[i*2+j]]
                se = pd.Series(data,index =['spot','spot_area','distance','flux'] )
                df = df.append(se,ignore_index = True)
        # flux和ix[5]数据的相对比值，假设不同构型下能谱不变      
        df['unit_ref'] = df['flux']/df.iloc[5]['flux']
        df['unit_area'] = df['spot_area']/df.iloc[5]['spot_area']
        df['flux'] = df['flux']*self.power/100.0
        self.flux_option = df.copy()
    
    def read_raw_da(self):
        # 读取原始能谱数据 ： 60mm 76.6m 100KW 双束团
        filename = r'D:\en.txt'
        da = pd.read_table(filename,names = ['en_start','en_end','flux'])    
        # 原始数据为双发模式，实际实验为单发模式，所以flux除以2
        da['flux'] = da['flux']/2
        # 加速器功率引起的flux变化
        da['flux'] = da['flux']*self.power/100.00
        self.raw_data = da.copy()
            
    def flux_pluse(self,numb):
        # 将能谱转为单个脉冲的飞行时间谱  
        flux_profile = self.flux_option.iloc[numb]
        da = self.raw_data.copy()
        distance = float(flux_profile['distance'])
        ## da['distance'] = flux_profile['distance'],# /s 变为 /pulse
        # /MeV/cm2/pulse
        da['flux'] = da['flux']*flux_profile['unit_ref']/self.Freq
        # /cm2/pulse 一个能量切片里的flux
        da['flux_area'] = (da['en_end'] - da['en_start'])*da['flux']
        da['time_start'] = da['en_start'].map(lambda x: flight_t(x,flight_length = distance))
        da['time_end'] = da['en_end'].map(lambda x: flight_t(x,flight_length = distance))      
        da['time_section'] = da['time_start']-da['time_end']
        # 该pluse产生的中子，飞到distance时的flux
        # /cm2/ns,时间切片和能量切片对应，总的flux_area不变，即time_start到time_end期间，该值不变
        da['flux_time'] = da['flux_area']/da['time_section']
        # 能通量
        da['en_flux'] = da['en_start']*da['flux']
        da['energy_flux'] = (da['en_start']+da['en_end'])*da['flux_time']/2
        flux_profile['data'] = da.copy()
        return flux_profile

# 选择特定能量中子对应的注量
def select_flux_from_en(en,flux_numb):
    # select single energy data
    da = CSNS_source.flux_pluse(flux_numb)['data'].copy()
    junk = da.loc[en <= da['en_end']]
    return junk.iloc[0]
    
# 源信息全局变量
CSNS_source = Neutron_source(power=25)
###################################################################################
###################################################################################
# 飞行时间谱可视化
def plot_section_spectrum(da,index,ax,plot_option='plot',fill = True,color='b',alpha=0.3):
    # 通用多道谱可视化
    # plot, semilogx, semilogy, loglog
    # index : start_x, end_x, height
    plot_command = 'ax.'+plot_option+'(x,y,color = color,alpha=alpha)' 
    da = da.reset_index()
    for i in range(len(da)):
        x = da.iloc[i][index[0]],da.iloc[i][index[1]]
        y = da.iloc[i][index[2]],da.iloc[i][index[2]]
        eval(plot_command)
        if fill is True:
            ax.fill_between(x,y,facecolor=color,alpha=alpha)
    for i in range(len(da)-1):
        x = da.iloc[i][index[1]],da.iloc[i][index[1]]
        y = da.iloc[i][index[2]],da.iloc[i+1][index[2]]
        eval(plot_command)

def select_energy(en,da):
    # select single energy data
    junk = da.loc[en <= da['en_end']]
    return junk.iloc[0]

def neutron_spectrum(nn):
    # 能谱
    flux_profile = CSNS_source.flux_pluse(nn)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=12)    
    plt.yticks(fontsize=12)
    da = flux_profile['data']
    plot_section_spectrum(da,['en_start','en_end','flux'],ax,plot_option='loglog')
    #ax.set_ylabel(u'Neutron flux (/MeV/$cm^2$/pluse)',fontsize=12)    
    ax.set_ylabel(u'中子注量 (/MeV/$cm^2$/pluse)',fontsize=12)    
    ax.set_xlabel(u'中子能量 (MeV)',fontsize=12)
    ax.text(0.5,0.9,str(CSNS_source.power)+u"KW 加速器能量",transform = ax.transAxes,fontsize=12)
    text = u'@'+str(flux_profile['distance'])+'  @'+flux_profile['spot']
    ax.text(0.5,0.85,text,transform = ax.transAxes,fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def time_spectrum(nn):
    # 飞行时间谱
    flux_profile = CSNS_source.flux_pluse(nn)
    distance = flux_profile['distance']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(axis='y')
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=12)    
    plt.yticks(fontsize=12)
    da = flux_profile['data']
    plot_section_spectrum(da,['time_start','time_end','flux_time'],ax,plot_option='semilogx')
    ax.set_ylabel(u'中子注量率 (/ns/$cm^2$)',fontsize = 12) 
    ax.set_xlabel(u'飞行时间 (ns)',fontsize = 12)    
    # neutron energy label
    energy_label = [0.001,0.01,0.1,1,10,100]
    x_start = 0.6
    label_control = 1
    height_limit = max(da['flux_time'])*1.1
    for x in energy_label:
        xx = [relative_t(x,flight_length=distance),relative_t(x,flight_length=distance)]
        yy =[select_energy(x,da)['flux_time'],height_limit]
        if label_control == 1:
            ax.semilogx(xx,yy,'--',color='g',label = u'中子能量(MeV)',linewidth = 2)
        else:
            ax.semilogx(xx,yy,'--',color='g',linewidth = 2)
        ax.text(x_start,1.05,str(x),transform = ax.transAxes)
        x_start = x_start - 0.1
        label_control = label_control*0   
    # flux info
    ax.text(0.65,0.25,str(CSNS_source.power)+u"KW 加速器能量",transform = ax.transAxes,fontsize=12)
    text = u'飞行距离: '+str(flux_profile['distance'])+'m'
    ax.text(0.65,0.20,text,transform = ax.transAxes,fontsize=12)
    text = u'束斑尺寸: '+flux_profile['spot']
    ax.text(0.65,0.15,text,transform = ax.transAxes,fontsize=12)
    # gamma
    t_gamma = distance*10**9/light_c
    ax.semilogx([t_gamma,t_gamma],[0,height_limit],'-',color='y',label = '$\gamma$ ('+ '%.2f' % t_gamma +'ns)',linewidth=3)
    ax.text(0.04,1.05,u"$\gamma$",transform = ax.transAxes)
    plt.legend(fontsize=12,frameon=True,facecolor='grey',framealpha = 0.5)
    plt.show() 