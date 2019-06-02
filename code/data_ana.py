# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import target_model as tm
import tof
import matplotlib.pylab as plt
import csns_read_nightly as cra

def en_from_peakamp(x):
    # 道值能量关系
    A = (5.805-5.275)/(1334.27418 - 1200.32056)
    B = 5.275 - 1200.32056*A
    return x*A + B

def cali_peak_ana():
    store = pd.HDFStore( r'D:\root\11712.h5')
    datafile_path = r'F:\single\\'
    cra.file_save(1,store,run_numb = 11712,datafile_path = datafile_path,board_id = 0,key='data',save = True)
    junk = store.select('data')
    cra.file_ana(junk,store,baseline_points = 80,polarize = -1,datafile_path = datafile_path,method = 'peak_ana_simple',save = True)
    store.close()

def cali():
    store = pd.HDFStore( r'D:\root\11712.h5')
    data = store.select('data_ana','board_id = 2 & peak_amp <2000 & peak_amp>700')
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    test = plt.hist2d(data['peak_time'],data['peak_amp'],bins=100)
    cbar = plt.colorbar()
    cbar.set_label(u'计数',fontsize = 14)
    plt.xlabel(u'时间(ns)',fontsize = 14)
    plt.ylabel(u'幅度(道值)',fontsize = 14)
    plt.show()
    
def cali1():
    store = pd.HDFStore( r'D:\root\11712.h5')
    data = store.select('data_ana','board_id = 2 & peak_amp <2000 & peak_amp>700')
    fig = plt.figure()
    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    test = plt.hist(data['peak_amp'],bins=100)
    plt.xlabel(u'幅度(道值)',fontsize = 14)
    plt.show()


def read_theory():
    filename = r'E:\\PIN20.txt'
    ef_info = pd.read_csv(filename,skiprows= 2,sep = '\s+',escapechar='*')
    data = ef_info.iloc[:72]
    data = data.astype('float')
    return data

def read_exp():
    store = pd.HDFStore( r'D:\root\single_pulse.h5')
    data_plot = store.select('data_ana','board_id = 2')
    data_plot['p_en_new'] = data_plot['peak_amp'].map(lambda x : en_from_peakamp(x))
    store.close()
    return data_plot


def ch2_raw():
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    store = pd.HDFStore( r'D:\root\single_pulse.h5')
    data_plot = store.select('data_ana','board_id = 2 & tof_time <6000 & peak_amp <1500 & peak_time > -1400')
    plt.hist2d(data_plot['peak_time'],data_plot['peak_amp'],bins=200,vmin=0, vmax=240)
    cbar = plt.colorbar(extend='max')
    
    plt.xlabel(u'单粒子脉冲峰值时刻(ns)',fontsize = 14)
    plt.ylabel(u'道值',fontsize = 14)
    store.close()

def ch2_gamma():
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    store = pd.HDFStore( r'D:\root\single_pulse.h5')
    data_plot = store.select('data_ana','board_id = 2 & peak_time < -200 & peak_amp <300 & peak_time > -1600' )
    plt.hist2d(data_plot['peak_time'],data_plot['peak_amp'],bins=100,vmin=0, vmax=1300)
    cbar = plt.colorbar(extend='max')
    
    plt.xlabel(u'单粒子脉冲峰值时刻(ns)',fontsize = 14)
    plt.ylabel(u'道值',fontsize = 14)
    store.close()
    
def ch2_gamma_hist():
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    store = pd.HDFStore( r'D:\root\single_pulse.h5')
    data_plot = store.select('data_ana','board_id = 2 & peak_time < -200 & peak_amp <300 & peak_time > -1600' )
    plt.hist(data_plot['peak_time'],bins=260)    
    plt.xlabel(u'单粒子脉冲峰值(ns)',fontsize = 14)
    plt.ylabel(u'统计粒子数',fontsize = 14)
    store.close()

def ch2():
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    # 实验值
    data_plot = pd.read_pickle(r'D:\root\ch2_single_new.pkl')
    # 杨教授理论值
    data = read_theory()
    
    junk = plt.hist2d(data_plot['n_en'],data_plot['p_en_new'],bins=200,vmin=0, vmax=400)
    plt.plot(data['En(MeV)'],data['Ep(MeV)'],'o-',color='red',alpha=0.5,label = u'理论模型')
    cbar = plt.colorbar(extend='max')
    cbar.set_label(u'反冲质子计数',fontsize = 14)

    #cbar.set_clim([0,200],extend='upper')
    plt.xlabel(u'入射中子能量(MeV)',fontsize = 14)
    plt.ylabel(u'反冲质子能量(MeV)',fontsize = 14)
    plt.legend(fontsize=14,frameon=True,facecolor='white',framealpha = 0.7)

    plt.show()

def ch2_rate():
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    data_plot = pd.read_pickle('df.pkl').iloc[:-1]
    data = read_theory()

    plt.plot(data['En(MeV)'],data['Rate'],'o-',color='red',alpha=0.5,label = u'理论模型')
    plt.plot(data_plot['n_en'],data_plot['p_numb_exp']/data_plot['p_numb_theory'],'o-',color='blue',alpha=0.5,label = u'实验结果')
    plt.legend(fontsize=14,frameon=True,facecolor='grey',framealpha = 0.7)
    plt.xlabel(u'入射中子能量(MeV)',fontsize = 14)
    plt.ylabel(u'反冲质子逃逸率',fontsize = 14)
    plt.show()


import target_model as tm
import tof
def u_signal_numb_calc(en_section=[0.2,1]):
    u1 = tm.U_slab()
    u1.cs_database(en_section[0],en_section[1])
    pin = tm.PIN()
    a = tm.Detector_Ana(u1,pin,flux_numb = 4,counts_all = 4500,en_section = en_section)
    a.cs['flux'] = a.cs['en'].map(lambda x : tof.select_flux_from_en(x,4)['flux'])
    en_section = a.cs['en'].diff()[1:]
    n_signal = en_section.values*a.cs['flux'][1:].values*a.cs['detec_eff'][1:].values
    n_signal_pulse = n_signal.sum()
    return n_signal_pulse

def ch2_signal_numb_calc(en_section=[0.2,1]):
    # 每个加速器pulse在en_section内产生的反冲质子个数，默认单束团，25KW(相对100KW而言)
    ch2 = tm.CH2_slab(pd = 2.432,phi =20)
    ch2.cs_database(en_section[0],en_section[1])
    pin = tm.PIN(phi =20)
    a = tm.Detector_Ana(ch2,pin,flux_numb = 4,counts_all = 4500,en_section = en_section)
    a.cs['flux'] = a.cs['en'].map(lambda x : tof.select_flux_from_en(x,4)['flux'])
    en_section = a.cs['en'].diff()[1:]
    n_signal = en_section.values*a.cs['flux'][1:].values*a.cs['detec_eff'][1:].values
    n_signal_pulse = n_signal.sum()
    return n_signal_pulse
    
def ch2_signal_numb_calc_new(en_section=[1,2],theta_slab = 90):
    # 每个加速器pulse在en_section内产生的反冲质子个数，默认单束团，25KW(相对100KW而言)
    ch2 = tm.CH2_slab(pd = 2.432,phi =20,theta_slab = theta_slab)
    ch2.cs_database(en_section[0],en_section[1])
    pin = tm.PIN(phi =20)
    a = tm.Detector_Ana(ch2,pin,flux_numb = 4,counts_all = 4500,en_section = en_section)
    a.cs['flux'] = a.cs['en'].map(lambda x : tof.select_flux_from_en(x,4)['flux'])
    en_section = a.cs['en'].diff()[1:]
    n_signal = en_section.values*a.cs['flux'][1:].values*a.cs['detec_eff'][1:].values
    n_signal_pulse = n_signal.sum()
    return n_signal_pulse

'''
def neutron_flux_pulse(en_section=[0.2,1]):
    # 每个加速器pulse在en_section内的中子个数，默认单束团，25KW(相对100KW而言)
    ch2 = tm.CH2_slab(pd = 2.432,phi =20)
    ch2.cs_database(en_section[0],en_section[1])
    ch2.cs['flux'] = ch2.cs['en'].map(lambda x : tof.select_flux_from_en(x,4)['flux'])
    en_section = ch2.cs['en'].diff()[1:]
    n_neutron = en_section.values*ch2.cs['flux'][1:]
    neutron_all = n_neutron.sum()
    return neutron_all
'''
def neutron_flux_pulse(en_section):
    # 每pluse的注量 /cm2, 加速器功率12.5KW，单束团
    data = tof.CSNS_source.flux_pluse(4)['data']
    ind1 = data[data['en_start'] < en_section[0]].iloc[-1].name
    ind2 = data[data['en_end'] > en_section[1]].iloc[0].name
    junk = data.loc[ind1:ind2].copy()
    aaa = (junk['en_end'] - junk['en_start']).values
    aaa[0] = junk.iloc[0]['en_end'] - en_section[0]
    aaa[-1] =  en_section[1] -  junk.iloc[-1]['en_start']
    junk['en_len'] = aaa
    junk['flux_select'] = junk['flux']*junk['en_len']
    return junk['flux_select'].sum()

def neutron_pulse_flux_from_acc(en_section,utc_numb):
    # 返回某一时间段内，某一能量切片的flux /cm2/s
    store_proton = pd.HDFStore(r'D:\root\hdf\proton_flux.h5')
    data_proton = store_proton.select('data','utc_time > %i & utc_time < %i' %(utc_numb[0]-1,utc_numb[1]+1))
    group_proton = data_proton.groupby('utc_time')['unknown'].mean().copy()
    store_proton.close()
    flux1 = group_proton.mean()   # 每pulse平均质子打靶数
    flux2 = flux1*25*0.413        # 折算到每中子注量 /cm2/s
    flux3 = neutron_flux_pulse(en_section)  #按照10KW下2.2e6的flux2计算能区注量  /cm2/pulse
    rate = flux2/2.2e6            # 实际注量/理论注量
    return flux3*rate             # 10KW的实际每pulse注量  /cm2/pulse
    

def sensi_ch2(data,en_section,n_pulse):
    # 返回灵敏度，估算质子个数，实际质子个数。加速器功率10KW,单束团
    cc = data['p_en'].sum()*10**6*tof.unit_e[0]/3.62   # C
    flux = neutron_flux_pulse(en_section)*20/25*n_pulse # /cm2 
    return cc/flux,ch2_signal_numb_calc(en_section)*20/25*n_pulse,len(data)


class En_Slice():
    # 切片分析，针对反冲质子靶
    def __init__(self,en_section,data):
        self.en_section = en_section
        self.data = data
        self.data['p_en'] = self.data['peak_amp'].map(lambda x : en_from_peakamp(x))

    
    def plot_raw(self,bins = 100,vmax = 20):
        junk = plt.hist2d(self.data['n_en'],self.data['p_en'],bins=bins,vmax = vmax)
        cbar = plt.colorbar(extend= 'max')
    
    def plot_slice(self,top,bottom):
        self.top = top
        self.bottom = bottom
        self.top_line = [[self.en_section[0],top[0]],[self.en_section[1],top[1]]]
        self.bottom_line = [[self.en_section[0],bottom[0]],[self.en_section[1],bottom[1]]]
        plt.plot(self.en_section,[self.top_line[0][1],self.top_line[1][1]],'--',color = 'r')  
        plt.plot(self.en_section,[self.bottom_line[0][1],self.bottom_line[1][1]],'--',color = 'r')
    
    def select_slice(self):
        self.data['p_en_top'] = self.data['n_en'].map(lambda x : self.yy(x,self.top_line[0],self.top_line[1])) - self.data['p_en']
        self.data['p_en_bottom'] = self.data['p_en'] - self.data['n_en'].map(lambda x : self.yy(x,self.bottom_line[0],self.bottom_line[1]))
        data_select = self.data[self.data['p_en_top'] > 0].copy()
        self.data_slice = data_select[data_select['p_en_bottom'] >0]
    
    def sensi_ch2(self,n_pulse):
        # 灵敏度，估算质子个数，实际质子个数
        cc = self.data_slice['p_en'].sum()*10**6*tof.unit_e[0]/3.62   # C
        self.n_flux = neutron_flux_pulse(self.en_section)*20/25*n_pulse*0.7556 # /cm2  0.7556是单束团期间的中子注量权重，由加速器提供
        self.numb_p_calc = ch2_signal_numb_calc(self.en_section)*20/25*n_pulse
        self.numb_p_exp = len(self.data_slice)
        self.ss =  cc/self.n_flux
        
    def yy(self,x,p1,p2):
        return (x-p2[0])*(p1[1]-p2[1])/(p1[0]-p2[0])+p2[1]


class Slice(En_Slice):
    def __init__(self,en_section,store,key = 'data_correct',extra = ''):
        # extra里可以输入其他数据选择条件
        self.extra = extra
        data_plot = store.select(key,'n_en > %f & n_en <%f' % (en_section[0],en_section[1]) + extra)
        En_Slice.__init__(self,en_section,data_plot)
    
    def slice(self,slice_up,slice_down,bins = 100,vmax = 20):
        self.plot_raw(bins,vmax)
        self.plot_slice(slice_up,slice_down)
    
    def calc(self,n_pulse):
        self.select_slice()
        self.sensi_ch2(n_pulse)
        self.ep_mean = self.data_slice['p_en'].mean()

        index = ['n_en','p_en','p_numb_theory','p_numb_exp','sensitivity','neutron_numb',
                    'slice_top','slice_bottom','en_section','extra']
        self.output = pd.Series([np.mean(self.en_section),self.ep_mean,self.numb_p_calc,self.numb_p_exp,
                                self.ss,self.n_flux,self.top,self.bottom,self.en_section,self.extra],
                                index = index)
        print(self.numb_p_calc,self.numb_p_exp,self.ss)
    
    def save(self,df):
        return df.append(self.output,ignore_index=True)

class Slice_replay(Slice):
    def __init__(self,se):
        Slice.__init__(self,se.en_section,se.extra)
        self.slice(se.slice_top,se.slice_bottom)

def plot_rate():
    exp = pd.read_csv(r'D:\root\ch2_sensi.csv',index_col=0)
    theory =  read_theory()
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    plt.plot(theory['En(MeV)'],theory['Rate'],'o-',color='black',alpha=0.5,label = u'理论模型')
    plt.plot(exp['n_en'],exp['p_numb_exp']/exp['p_numb_theory'],'o-',color='blue',alpha=0.5,label = u'单束团实验')
    plt.legend(fontsize=14,frameon=True,facecolor='grey',framealpha = 0.7)
    plt.xlabel(u'入射中子能量(MeV)',fontsize = 14)
    plt.ylabel(u'反冲质子逃逸率',fontsize = 14)
    plt.show()
    
def plot_sensi():
    exp = pd.read_pickle(r'D:\root\ch2_single_new.pkl')
    theory =  read_theory()
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    plt.plot(theory['En(MeV)'],theory['Sn(A.cm^2.s)'],'o-',color='black',alpha=0.5,label = u'理论模型')
    plt.plot(exp['n_en'],exp['sensitivity']*1.05,'o-',color='blue',alpha=0.5,label = u'单束团实验')
    plt.legend(fontsize=14,frameon=True,facecolor='grey',framealpha = 0.7)
    plt.xlabel(u'入射中子能量(MeV)',fontsize = 14)
    plt.ylabel(u'灵敏度$(A.cm^2.s)$',fontsize = 14)
    plt.show()

class ET_Coin():
    # 能量道与时间道符合，只是同一个t0_id的合并在一起，还没有选区域
    # x : t 
    # y : e
    def __init__(self,data_e,data_t):
        test_e = data_e[['t0_id','file_numb','run_numb','wf_index','peak_amp','peak_time','address']].copy()
        data_coin = pd.merge(data_t,test_e,on =['t0_id','file_numb','run_numb'])
        data_coin['peak_time_diff'] = data_coin['peak_time_y'] - data_coin['peak_time_x']
        data_coin['peak_amp_divide'] = data_coin['peak_amp_y']/data_coin['peak_amp_x']
        self.data_coin = data_coin

    def save_to_store(self,store):
        store.append('data_coin',self.data_coin,format='table', data_columns=True,index = False,complevel=4, complib='blosc')      
    

class Dir_Files():
    # 统计数据文件夹中的文件信息
    import os
    def __init__(self,path):
        self.path = path
        junk = []
        for x,y,z in os.walk(path):
            junk.append(z)
        self.files = junk[0]
    
    def filename_data(self):
        self.data = pd.DataFrame()
        self.data['filename'] = self.files
        self.data['run_numb'] = self.data['filename'].map(lambda x: int(x.split('.')[0].split('-')[-3]))
        self.data['file_numb'] = self.data['filename'].map(lambda x: int(x.split('.')[0].split('-')[-1]))
        self.data['time_numb'] = self.data['filename'].map(lambda x : self.file_time(x))
        self.data['date'] = self.data['time_numb'].map(lambda x : cra.utc_time(x).split()[0])
        self.data['time'] =self.data['time_numb'].map(lambda x : cra.utc_time(x).split()[1])
        self.data['size'] = self.data['filename'].map(lambda x : os.path.getsize(self.path+'\\'+x))
    
    def file_time(self,filename):
        filename = self.path + '\\'+ filename
        junk = cra.CSNS_Read(filename)
        junk.read_file_header(index = 1)
        junk.close()
        return junk.file_header['file_open_time']
    
    def run_numb_ana(self):
        self.grouped = self.data.groupby(['run_numb'])
        self.run_numb_list = pd.DataFrame(self.grouped.size(),columns = ['file_numb_sum'])
        self.run_numb_list['file_numb_max'] = self.grouped['file_numb'].max().values
        self.run_numb_list['file_numb_missed'] = self.run_numb_list['file_numb_max'] - self.run_numb_list['file_numb_sum']
        
class DataFrame_Grouped_Slice():
    # 按照index_group进行分组
    def __init__(self,df,index_group = ['run_numb','file_numb','board_id','channel']):
        self.df = df
        self.index_group = index_group
        self.grouped = self.df.groupby(self.index_group)
        self.grouped_size = pd.DataFrame(self.grouped.size())
    
    def slice_df_index(self,index_slice = ['board_id','channel'],value_slice = [3,2]):
        # 按照index_slice进行索引切片
        dd = dict(zip(index_slice,value_slice))
        self.values = []
        for x in self.index_group:
            if x not in index_slice:
                self.values.append(slice(None))
            else:
                self.values.append(dd[x])
        self.values = tuple(self.values)
        self.slice_size = self.grouped_size.loc[self.values,:]
        self.slice_index = self.slice_size.index
        self.slice_index_len = len(self.slice_index)
    
    def select_df(self,i=0):
        # 选择self.slice_index中某一index对应的df]
        self.index_now = self.slice_index[i]
        self.data_now = self.grouped.get_group(self.index_now)

class Plot():
    def __init__(self):
        fig = plt.figure()
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        plt.xticks(fontsize=14)    
        plt.yticks(fontsize=14)
        
    def label(self,xlabel,ylabel):
        cbar = plt.colorbar(extend = 'max')
        plt.xlabel(xlabel,fontsize = 14)
        plt.ylabel(ylabel,fontsize = 14)
    
class U_coin(Plot):
    def __init__(self):
        Plot.__init__(self)

    def data_plot(self,store,key = 'data_coin',select_condition = '',x_name = 'peak_time_diff',
                    y_name = 'peak_amp_divide',xlabel = '',ylabel = '',**kargs):
        self.store = store
        self.key = key
        self.x_name = x_name
        self.y_name = y_name
        self.data_all = self.store.select(key,select_condition)
        self.data_all.reset_index(inplace = True)
        self.data_all.drop(['index'],axis=1,inplace =True)
        test = plt.hist2d(self.data_all[self.x_name],self.data_all[self.y_name],**kargs)
        self.label(xlabel,ylabel)

    def data_slice(self,xx,yy):
        # 画出被选出的
        self.xx = xx
        self.yy = yy
        plt.plot(xx,[yy[0],yy[0]],color = 'red')
        plt.plot(xx,[yy[1],yy[1]],color = 'red')
        plt.plot([xx[0],xx[0]],yy,color = 'red')
        plt.plot([xx[1],xx[1]],yy,color = 'red')
    
    def slice_select(self,remove = True):
        key_bk = self.key+'_backup'
        try:
            self.store.append(key_bk,self.data_all,format='table', data_columns=True,index = False,complevel=4, complib='blosc')
        except:
            self.store.remove(key_bk)
        self.select_condition = '%s > %f & %s < %f & %s > %f & %s < %f' % (self.x_name,self.xx[0],
                            self.x_name,self.xx[1],self.y_name,self.yy[0],self.y_name,self.yy[1])     
        self.data = self.store.select(key_bk,self.select_condition)
        if remove:
            self.store.remove(key_bk)
        
    def slice_delete(self):
        self.slice_select()
        self.data = self.data_all.drop(self.data.index)
    
    def save_to_store(self,key = True):
        if key is True:
            key = self.key + '_select'
        self.store.append(key,self.data,format='table', data_columns=True,index = False,complevel=4, complib='blosc')

class U_plot(Plot):
    def __init__(self):
        Plot.__init__(self)
        self.data_plot()
        self.label(u'时间(ns)',u'幅度')

    def data_plot(self):
        store = pd.HDFStore( r'D:\root\single_pulse.h5')       
        data_plot = store.select('data_coin','peak_time_diff > 1400 & peak_amp_divide <1.5  & peak_amp_divide >0.5 & peak_time_diff<1700')
        test = plt.hist2d(data_plot['peak_time_x'],data_plot['peak_amp_y'],bins=160,vmin = 0,vmax = 5)


class U_plot1(Plot):
    def __init__(self):
        Plot.__init__(self)
        self.data_plot()
        self.label(u'时间(ns)',u'幅度')

    def data_plot(self):
        store = pd.HDFStore( r'D:\root\single_pulse.h5')       
        data_plot = store.select('data_coin','peak_time_diff > 1400 & peak_amp_divide <1.5'
                    +  '& peak_amp_divide >0.5 & peak_time_diff<1700 & peak_time_x <800 & peak_amp_y < 600')
        test = plt.hist2d(data_plot['peak_time_x'],data_plot['peak_amp_y'],bins=160,vmin = 0,vmax = 20)

def ana_u_e():
    store = pd.HDFStore( r'D:\root\exp_pin3.h5')       
    aa = store.select('data','board_id = 3')
    cc = DataFrame_Grouped_Slice(aa)
    cc.slice_df_index(index_slice=['board_id'],value_slice=[3])
    for i in range(cc.slice_index_len):
        cc.select_df(i)
        cra.file_ana(cc.data_now,store,baseline_points = 1000,polarize = 1,datafile_path = r'F:\database\\',method = 'peak_ana_simple',save = True)
        
class U_slab_plot(Plot):
    def __init__(self):
        Plot.__init__(self)
        self.data_plot()

    def data_plot(self):
        data = pd.read_csv(r'D:\root\u235_correct.csv')
        data_plot = data[data['n_en'] > 0.2][data['n_en']<22].copy()
        test = plt.hist2d(data_plot['n_en'],data_plot['p_en'],bins = 70,vmax = 3)
        self.label(u'中子能量(MeV)',u'裂变碎片能量(MeV)')

    
    def data_slice(self,en_section):
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        plt.xticks(fontsize=14)    
        plt.yticks(fontsize=14)
        n_pluse = 675451.0
        # 双束团
        neutron_per_pluse = neutron_flux_pulse(en_section)*2*20.0/25.0
        data = pd.read_csv(r'D:\root\u235_correct.csv')
        data_plot = data[data['n_en'] > en_section[0]][data['n_en']<en_section[1]].copy()
        test = plt.hist(data_plot['p_en'],bins = 50,alpha=0.6,label = str(en_section))
        cc = data_plot['p_en'].sum()*10**6*tof.unit_e[0]/3.62   # C
        self.ss =  cc/n_pluse/neutron_per_pluse
        self.nn = len(data_plot)

        
def plot3():
    data_exp =  pd.read_pickle(r'D:\root\ch2_single_new.pkl')
    data_exp =  data_exp.sort_values('n_en')

    data = read_theory()
    store = pd.HDFStore(r'D:\root\single_pulse.h5')
    data_plot = store.select('data_correct','n_en < 21 & p_en <7.5')
    data_plot['p_en'] = data_plot['peak_amp'].map(lambda x : en_from_peakamp(x))
    
    fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.xticks(fontsize=14)    
    plt.yticks(fontsize=14)
    
    
    junk = plt.hist2d(data_plot['n_en'],data_plot['p_en'],bins=200,vmin=0, vmax=70)
    plt.plot(data['En(MeV)'],data['Ep(MeV)'],'o-',color='red',alpha=0.5,label = u'质子平均能量(理论)')
    plt.plot(data_exp['n_en'],data_exp['p_en'],'o-',color='white',alpha=0.5,label = u'质子平均能量(实验)')

    cbar = plt.colorbar(extend='max')
    cbar.set_label(u'反冲质子计数',fontsize = 14)
    
    #cbar.set_clim([0,200],extend='upper')
    plt.xlabel(u'入射中子能量(MeV)',fontsize = 14)
    plt.ylabel(u'反冲质子能量(MeV)',fontsize = 14)
    plt.legend(fontsize=14,frameon=True,facecolor='white',framealpha = 0.7)

    store.close()
    plt.show()  


def curve_fit_gaussian(xdata,ydata,sigma,parm='default'):
#------------------------------------------------------------------------------------------------------
    def gaussianm(x,*parm):
        peak_numb = (len(parm)-1)/3
        junk = parm[-1]
        for i in range(peak_numb):
            junk = junk + parm[i*3]*np.exp(-np.power(x-parm[i*3+1],2)/(2*np.power(parm[i*3+2],2)))
        return junk
#------------------------------------------------------------------------------------------------------        
    from scipy.optimize import curve_fit
    # find default p0
    # a: I.I,  b: x[max(y)], c: FMWH, d: bg, e: bg slope
    d = min(ydata)
    b = xdata[ydata == max(ydata)][0]
    junk = abs(ydata-max(ydata)/2)
    xl = xdata[xdata < b][junk[xdata < b] == min(junk[xdata < b])][0]
    xh = xdata[xdata > b][junk[xdata > b] == min(junk[xdata > b])][0]
    c = abs((xh-xl)/2)
    a = abs(c*2*max(ydata))
    if parm == 'default':
        parm = [a,b,c,d]
#------------------------------------------------------------------------------------------------------        
    popt, pcov = curve_fit(gaussianm, xdata, ydata,p0 = parm,sigma=sigma)
    margin = (max(xdata)-min(xdata))/10
    x1 = np.linspace(min(xdata)-margin,max(xdata)+margin,100)
    plt.errorbar(xdata,ydata,yerr = sigma,fmt='o')
    perr = np.sqrt(np.diag(pcov))
    n_peak = (len(parm)-1)/3
    label_text_data = []
    label_text_text = ''
    for i in range(len(popt)):
        label_text_data.append(popt[i])
        label_text_data.append(perr[i])
    for i in range(n_peak):
        label_text_text = label_text_text + 'I.I : {'+str(i*6+0)+':.3f}({'+str(i*6+1)+':.3f})\n' \
                                            + 'Peak : {'+str(i*6+2)+':.3f}({'+str(i*6+3)+':.3f})\n' \
                                            + 'FMWH : {'+str(i*6+4)+':.3f}({'+str(i*6+5)+':.3f})\n'+'-'*30+'\n'
    label_text_text = label_text_text + 'BG : {'+str(len(popt)*2-2)+':.3f}({'+str(len(popt)*2-1)+':.3f})'
    label_text = label_text_text.format(*label_text_data)

    plt.plot(x1,gaussianm(x1,*popt),'-',label=label_text)
    plt.legend()  
    plt.show()
    return popt,perr   

def curve_fit(data,xran='all',parm='default'):
    # find default p0
    # a: I.I,  b: x[max(y)], c: FMWH, d: bg, e: bg slope
    xdata = data['x']
    ydata = data['y']
    sigma = data['dy']
    if xran != 'all':
        index_select = xdata[xdata > xran[0]][xdata < xran[1]].index
        xdata = xdata[index_select]
        ydata = ydata[index_select]
        sigma = sigma[index_select]
    xdata = xdata.values
    ydata = ydata.values
    sigma = sigma.values
    fit_out = curve_fit_gaussian(xdata,ydata,sigma,parm = parm)
    plt.show()
    return fit_out
    