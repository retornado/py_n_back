# -*- coding: utf-8 -*-
import struct
import time
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import binascii

# 调整结构，以t0包为输入,默认数据文件每次更新都是以t0数据包为最小单位
################################################################################################
# CSNS数据读取基础类
class CSNS_Read():
    def __init__(self,filename):
        self.filename = filename
        self.file_numb = int(self.filename.split('.')[0].split('-')[-1])
        self.run_numb = int(self.filename.split('.')[0].split('-')[-3])        
        self.file = open(filename,'rb')
        self.file_index = ['header_len','version','run_number','run_start_time',
                  'file_open_time']
        self.t0_index =  ['t0_len','t0_id','numb_signal_package','tcm_board_id',
                    'start_flag','tcm_t0_id','tcm_time','tcm_time_correct']
        self.fdm_index = ['board_id','signal_len','start_flag','detector_type',
                         'trigger','fdm_t0_id','tof_time','tof_time_correction','channel']
        
    def read_file_header(self,index = 0):         
        # 20
        self.file_header = struct.unpack('5I',self.file.read(20))
        if index:
            self.file_header = dict(zip(self.file_index,self.file_header))
    
    def read_t0_header(self,index = 0):
        # 56
        self.t0_header = list(struct.unpack('4x3I8x',self.file.read(24)))
        signal_header = struct.unpack('I12xB4xB',self.file.read(22))
        # 单独存储raw，方便分析
        self.tcm_time_raw = self.file.read(9)
        tcm_time = raw_to_int(self.tcm_time_raw,reverse=False)
        tcm_time_correct = struct.unpack('B',self.file.read(1))
        self.t0_header.extend(signal_header)
        self.t0_header.append(tcm_time)
        self.t0_header.extend(tcm_time_correct)
        if index:
            self.t0_header = dict(zip(self.t0_index,self.t0_header))
        
    def read_fdm_header(self,index = 0):
        # 48
        signal_header = struct.unpack('2I8xB4xB',self.file.read(22))
        trigger_time = struct.unpack('>H8xB',self.file.read(11))
        # 可能为负数，详见tof_time负数问题解析 
        junk = self.file.read(3)  
        tof_time = struct.unpack('>i',junk+'\0')[0] >> 8
        tof_time = tof_time*8
        junk1 = struct.unpack('B10xB',self.file.read(12))
        self.fdm_header = list(signal_header) + list(trigger_time)+[tof_time]+list(junk1)
        if index:
            self.fdm_header = dict(zip(self.fdm_index,self.fdm_header))
        
    def read_fdm_data(self,index = 0):
        if index:
            N = self.fdm_header['signal_len'] - 48
        else:
            N = self.fdm_header[1] - 48
        self.wf_data = struct.unpack('>'+str(N/2)+'H',self.file.read(N))
    
    def skip_fdm_data(self):
        N = self.fdm_header[1] - 48
        junk = self.file.read(N)
        
    def plot_wf(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.plot(self.wf_data)
   
    def tcm_time_ana(self):
        self.tcm_time_s =  struct.unpack('>I',self.tcm_time_raw[1:5])[0]
        self.tcm_time_ns = struct.unpack('>I',self.tcm_time_raw[5:])[0]    
         
    def close(self):
        self.file.close() 

################################################################################################
# 基础类子类
# 1 . 只读取发次信息，速度较快
class CSNS_T0_Index(CSNS_Read):
    def __init__(self,filename):
        CSNS_Read.__init__(self,filename)
        self.creat_t0_list()
            
    def creat_t0_list(self):
        self.read_file_header(index=True)
        data = []
        while True:
            try:
                self.read_t0_header()
                data.append(self.t0_header)
                self.file.read(self.t0_header[0]-56)
            except:
                self.file.close()
                break
        self.t0_list = pd.DataFrame(np.array(data),columns = self.t0_index)
        self.t0_list['run_numb'] = self.run_numb
        self.t0_list['file_numb'] = self.file_numb
        # addition
        self.tcm_time_ana()
        self.t0_list['tcm_time_s'] = self.tcm_time_s

    
################################################################################################
# 2. 读取发次和波形信息，形成波形索引
class CSNS_Index(CSNS_Read):
    def __init__(self,filename):
        CSNS_Read.__init__(self,filename)
        self.creat_index()
    
    def creat_index(self):
        self.read_file_header(index=True)
        data_t0 = []
        data_wf = []
        while True:
            try:            
                self.read_t0_header()
                data_t0.append(self.t0_header)
                for i in range(self.t0_header[2]-1): 
                    self.read_fdm_header()
                    # add t0_id
                    self.fdm_header.append(self.t0_header[1])
                    self.skip_fdm_data()
                    data_wf.append(self.fdm_header)
            except:
                self.file.close()
                break
        self.t0_list = pd.DataFrame(np.array(data_t0),columns = self.t0_index)
        self.t0_list['t0_index'] = self.t0_list.index
        columns_name = self.fdm_index+['t0_id']
        self.wf_list = pd.DataFrame(np.array(data_wf),columns = columns_name)  
        self.wf_list['signal_len_cumsum'] = self.wf_list['signal_len'].cumsum()
        self.wf_list['wf_index'] = self.wf_list.index
        self.data = pd.merge(self.wf_list,self.t0_list,on = 't0_id')
        self.data['run_numb'] = self.run_numb
        self.data['file_numb'] = self.file_numb
        self.data['address'] = 20 + (self.data['t0_index']+1)*56 + self.data['signal_len_cumsum'] + 48 - self.data['signal_len']
          
    def store_data(self,key='data'):
        store_file = r'D:\root\%i.h5'%(self.run_numb)
        store = pd.HDFStore(store_file)
        if '/'+key in store.keys():
            if self.file_numb not in store[key]['file_numb'].unique():
                store.append(key,self.data,format='table', data_columns=True,index = False)
                print(self.filename + ' stored successful !!')
            else:
                print(self.filename + ' already in the store !!')   
        else:
            store.append(key,self.data,format='table', data_columns=True,index = False)
            print(self.filename + ' stored successful !!')
        store.close()
################################################################################################
class WF_ANA_File():
    def __init__(self,df,datafile_path=r'F:\11683\\'):
        #  输入为单个文件的df,从上一个类中产生
        self.datafile_path = datafile_path
        self.run_numb = df['run_numb'].iloc[0]
        self.file_numb = df['file_numb'].iloc[0]
        self.filename = datafile_path + 'daq-%5i-NORM-%02i.raw' % (self.run_numb,self.file_numb)
        self.df = df
        self.len =  len(df)   
        # wf间要跳过的长度
        address = self.df['address'].diff().values[1:]
        address = address - self.df['signal_len'].values[:-1]+48
        address = address.astype('int64')
        self.address = [self.df['address'].values[0]] + list(address)
        
    def analyse(self,func,**parms):
        # 将函数作为参数，传递给analyse函数
        self.peak_list = []
        f = open(self.filename,'rb')
        for i in range(self.len):
            # 读除wf_data
            f.read(self.address[i])
            self.nn = self.df['signal_len'].iloc[i]-48
            wf_signal = struct.unpack('>'+str(int(self.nn/2))+'H',f.read(self.nn)) 
            tof_time = self.df['tof_time'].iloc[i] + np.array(range(int(self.nn/2)))
            self.wf_data = pd.Series(wf_signal,index = tof_time)
            # 处理wf_data
            func(**parms)
            self.peak_list.append(self.data_line)
        f.close()
        self.data_out = pd.DataFrame(np.array(self.peak_list),columns=self.index_name)
        
    def peak_ana_simple(self,baseline_points=200,polarize=1):
        # 峰值强度
        self.index_name = ['peak_amp','peak_time','baseline']
        baseline = np.mean(self.wf_data[:baseline_points])
        if polarize == 1:
            peak_amp = self.wf_data.max() - baseline
            peak_posi = self.wf_data.idxmax()
        else:
            peak_amp = baseline - self.wf_data.min()
            peak_posi = self.wf_data.idxmin()
        self.data_line = [peak_amp,peak_posi,baseline]
    
    def peak_ana_ii(self,baseline_points=200,polarize=1):
        # 积分强度
        self.index_name = ['peak_II','peak_amp','peak_time','baseline']
        baseline = np.mean(self.wf_data[:baseline_points])
        if polarize == 1:
            peak_amp = self.wf_data.max() - baseline
            peak_posi = self.wf_data.idxmax()
            peak_ii = self.wf_data.sum()-baseline*self.nn
        else:
            peak_amp = baseline - self.wf_data.min()
            peak_posi = self.wf_data.idxmin()
            peak_ii = baseline*self.nn - self.wf_data.sum()            
        self.data_line = [peak_ii,peak_amp,peak_posi,baseline]
        
    def peak_ana_cfd(self,fraction = 0.5,n_trans = 10,base_aver = 750,n_aver =40,threld = 0.5,polarize=1):
        # 恒比定时方法
        # 但如果有异常波形信号，容易报错
        self.index = ['peak_amp','cfd_time','baseline']
        data = pd.DataFrame(np.array([self.wf_data.index,self.wf_data.values]).transpose(),
                                 columns = ['tof_time','input_signal'])
        if not polarize:
            data = data*-1
        baseline = np.mean(data['input_signal'][:base_aver])
        data['baseline_constant'] = 0
        data['input_signal'] = data['input_signal'] - baseline
        peak_amp = max(data['input_signal'])

        # 恒比定时处理后的信号
        data['cf'] = data['input_signal']*fraction
        data['translate'] = list(data['cf'][:n_trans]) + list(data['input_signal'][:-n_trans])
        data['inverse_translate'] = data['translate']*(-1)
        data['cfd'] = data['cf']+data['inverse_translate']
        
        # 找过零点
        data['index_new'] = data.index//n_aver
        grouped = data.groupby(data['index_new'])
        counts = grouped.sum()['cfd']
        thre = max(data['cfd'])*threld*n_aver
        index_start = (counts[counts>thre].index[0]+1)*n_aver  
        junk2 = data['cfd'][index_start:]
        cfd_time = data['tof_time'][junk2[junk2<0].index[0]]
        
        if not polarize:
            baseline = baseline*-1
        self.data_line = [peak_amp,cfd_time,baseline]

################################################################################################
# 针对单个波形的画图
class WF_ANA():
    def __init__(self,wf,plot = True,datafile_path = r'F:\database\\'):
        self.wf = wf.astype('int64')
        self.run_numb = self.wf.run_numb
        self.datafile_path =  datafile_path
        self.read_single_wf_data(plot)
    
    def read_single_wf_data(self,plot):
        # se为pd.Series格式
        filename = self.datafile_path + 'daq-%5i-NORM-%02i.raw' % (self.wf.run_numb,self.wf.file_numb)
        f = open(filename,'rb')
        f.read(self.wf.address)
        nn = self.wf.signal_len-48
        wf_signal = struct.unpack('>'+str(int(nn/2))+'H',f.read(nn)) 
        f.close()
        tof_time = self.wf.tof_time + np.array(range(int(nn/2)))
        self.wf_data = pd.Series(wf_signal,index = tof_time)
        if plot:
            self.plot_wf()
        
    def peak_ana_simple(self,baseline_points=200,polarize=1):
        self.index_name = ['peak_amp','tof_time','baseline']
        baseline = np.mean(self.wf_data[:baseline_points])
        if polarize == 1:
            peak_amp = self.wf_data.max() - baseline
            peak_posi = self.wf_data.idxmax()
        else:
            peak_amp = baseline - self.wf_data.min()
            peak_posi = self.wf_data.idxmin()
        self.data_line = [peak_amp,peak_posi,baseline]
         
    def plot_wf(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('time (ns)')
        text = 't0_id : %i , board_id : %i , channel : %i, wf_index : %i, file_numb : %i' % (self.wf['t0_id'],
                self.wf['board_id'],self.wf['channel'],self.wf['wf_index'],self.wf['file_numb'])
        ax.plot(self.wf_data.index,self.wf_data.values)
        ax.text(0.1,1.05,text,transform = ax.transAxes,fontsize=12)
        
################################################################################################
# 画出一个t0包信息 
class Channel_Ana():
    def __init__(self,t0_data,datafile_path = r'F:\database\\'):
        self.data = t0_data
        self.datafile_path = datafile_path

    def read_single_channel_data(self,df):
        # df先按照tof_time重排
        se = df.iloc[0]
        filename = self.datafile_path + 'daq-%5i-NORM-%02i.raw' % (se.run_numb,se.file_numb)
        f = open(filename,'rb')
        address = [se['address']] + list(df['address'][1:].values - df['address'][:-1].values - df['signal_len'][:-1].values + 48)  
        wf_signal = []
        tof_time = []
        for x in range(len(df)):
            f.read(address[x])
            nn = df.iloc[x]['signal_len']-48
            wf_signal.extend(struct.unpack('>'+str(nn/2)+'H',f.read(nn)))
            tof_time.extend(df.iloc[x]['tof_time'] + np.array(range(nn/2)))
        f.close()
        data = pd.Series(wf_signal,index=tof_time)
        data = data.sort_index()
        return data
        
    def plot(self):
        fig = plt.figure()
        grouped = self.data.groupby(['board_id','detector_type','channel'])
        ind = grouped.size().index
        n_ind = len(ind)
        for i in range(n_ind):
            ax = fig.add_subplot(n_ind,1,i+1)
            bb = grouped.get_group(ind[i])
            data = self.read_single_channel_data(bb) 
            ax.semilogx(data.index,data.values)
            ax.set_xlabel('TOF time (ns)')    
     
def condition_select_df(df,index_name,index):
    # index_name和index都以list方式输入，哪怕只有一个元素
    grouped = df.groupby(index_name)
    if len(index) >1:
        index = tuple(index)
    else:
        index  = index[0]
    return  grouped.get_group(index)

def raw_to_bin(line,reverse = True):
    # read bytes to binary
    # reverse = True ---  <
    numb = len(line)*8
    if reverse:
        line = line[::-1]
    a =  bin(int(binascii.b2a_hex(line),16))[2:]
    return '0'*(numb-len(a)) + a

def raw_to_int(line,reverse = True):
    # read bytes to int
    # reverse = True ---  <
    # 这里处理出来都是无符号数
    if reverse:
        line = line[::-1]
    return int(binascii.b2a_hex(line),16)
    	
# UTC time to string
def utc_time(time_numb):
    ltime = time.localtime(time_numb)
    return time.strftime("%Y-%m-%d %H:%M:%S",ltime)

import datetime
def utc_to_numb(utc_time_str,utc_format='%Y-%m-%d %H:%M:%S'):
    utc_dt = datetime.datetime.strptime(utc_time_str, utc_format)       #讲世界时间的格式转化为datetime.datetime格式
    return int(time.mktime(utc_dt.timetuple()))                       #返回utc时间戳
        
def select_run_numb_package(run_numb):
    datafile_path =  r'D:\root\\'
    return pd.HDFStore(datafile_path + str(run_numb)+'.h5')

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

def file_save(file_numb,store,run_numb = 11696,datafile_path = r'F:\database\\',board_id = 0,key='data',save = True):
    filename = datafile_path + 'daq-%i-NORM-%02i.raw' % (run_numb,file_numb)
    a = CSNS_Index(filename)
    if board_id == -1:
        data = a.data
    elif board_id == 0:
        data = a.data[a.data['board_id'] != 13]    #不存储Li-Si数据
    else:
        data = a.data[a.data['board_id'] == board_id]
    if save:
        if len(data) > 0:
            store.append(key,data,format='table', data_columns=True,index = False,complevel=4, complib='blosc')

def file_ana(junk,store,baseline_points = 1000,polarize = 1,datafile_path = r'F:\11695\\',method = 'peak_ana_simple',save = True):
    # 可自定义具体的分析函数
    peak_e = WF_ANA_File(junk,datafile_path=datafile_path)
    eval('peak_e.analyse(peak_e.%s,baseline_points=baseline_points,polarize=polarize)'%(method))
    aaa = np.array(peak_e.peak_list).T
    n_index_name = len(peak_e.index_name)
    for i in range(n_index_name):
        junk[peak_e.index_name[i]] = aaa[i]
    if save:
        store.append('data_ana',junk,format='table', data_columns=True,index = False,complevel=4, complib='blosc')
      

def plot_hist2d(file_numb,store,board_id = 3,channel = 2,set_clim = -1,bins=100):
    junk = store.select('data_ana','board_id = %i & channel = %i & file_numb = %i' %(board_id,channel,file_numb))
    test = plt.hist2d(junk['peak_time'],junk['peak_amp'],bins=bins)
    cbar = plt.colorbar()
    if set_clim != -1:
        cbar.set_clim(set_clim)

def t0_index_save(file_numb,store,key = 'data_t0',run_numb = 11699,datafile_path = r'F:\database\\'):
    filename = datafile_path + 'daq-%i-NORM-%02i.raw' % (run_numb,file_numb)
    a = CSNS_T0_Index(filename)
    store.append(key,a.t0_list,format='table', data_columns=True,index = False,complevel=4, complib='blosc')
    
    
    

