# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import target_model as tm
import tof
import matplotlib.pylab as plt
import csns_read_nightly as cra

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

def channel_slice(section,data):
    # 在多道谱中切片
    # 例如裂变截面数据到20MeV，考虑可以放宽数据标准，认为20MeV以上的和20MeV一样大
    try:
        # section可能不在data范围内
        ind1 = data[data['en_start'] <= section[0]].iloc[-1].name
        ind2 = data[data['en_end'] >= section[1]].iloc[0].name
    except :
        print('!!! Error：section out of data range!!!')
    junk = data.loc[ind1:ind2].copy()
    junk['en_start'].iloc[0] = section[0]
    junk['en_end'].iloc[-1] = section[1]
    return junk

def single_to_channel(data,name='n_en'):
    # 将散点数据变成多道数据，道宽取前后间隔的一半
    sec = list(data[name].diff().values[1:]/2)
    section = np.array([sec[0]] + sec + [sec[-1]])
    data['en_start']  = data[name] - section[:-1]
    data['en_end']  = data[name] + section[1:]
    data.drop([name],inplace= True,axis=1)
    return data

def merge_slice(data11,data22):
    # 合并两个多道谱，多次使用可合并任意多个多道数据
    data1 = data11.copy()
    data2 = data22.copy()
    # 构建合并后的分道 data3
    points1 = data1['en_start'].values
    points2 = data2['en_end'].values
    points = np.array(list(points1) + list(points2))
    points = np.unique(points)
    points.sort()
    data3 = pd.DataFrame()
    data3['en_start'] = points[:-1]
    data3['en_end'] = points[1:]
    # 丢掉en_end，等待合并
    data1.drop('en_end',axis = 1,inplace=True)
    data2.drop('en_end',axis = 1,inplace=True)
    # 数据合并与赋值
    data = pd.merge(data1,data3,how='outer')
    data = pd.merge(data,data2,how='outer')
    data.sort_values(by = 'en_start',inplace=True)
    data.fillna(method = 'ffill',inplace=True)
    # 数据清理：重编号，去重
    data.reset_index(inplace=True)
    data.drop(['index'],inplace=True,axis=1)
    data.drop_duplicates(inplace=True)
    return data

def plot_hist2d(file_numb,store,board_id = 3,channel = 2,set_clim = -1,bins=100):
    junk = store.select('data_ana','board_id = %i & channel = %i & file_numb = %i' %(board_id,channel,file_numb))
    test = plt.hist2d(junk['peak_time'],junk['peak_amp'],bins=bins)
    cbar = plt.colorbar()
    if set_clim != -1:
        cbar.set_clim(set_clim)

def phi_cs(en_section):
    # 求en_section内的中子产生的裂变碎片贡献
    # 截面
    cs = tm.cs_data_u235.copy()
    cs['en'] = cs['en']*10**(-6) # 最高到20MeV,扩展到1000MeV
    se = pd.Series([1000,1.9343],index= ['en','cs'])
    cs = cs.append(se,ignore_index=True)
    cs = single_to_channel(cs,name='en')
    cs_section = channel_slice(en_section,cs)
    
    # 能谱
    da1  = pd.read_pickle(r'D:\es1_flux.pkl')
    flux_section = channel_slice(en_section,da1)
    data_section = merge_slice(flux_section,cs_section)
    data_section.dropna(inplace=True)
    # flux乘以cs，分段求和
    data_section['junk'] = data_section['flux_normal']*data_section['cs']*(data_section['en_end']-data_section['en_start'])
    return data_section['junk'].sum()