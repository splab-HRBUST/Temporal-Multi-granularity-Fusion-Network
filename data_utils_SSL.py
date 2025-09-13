import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    #读取一个指定路径下的元数据文件，并将文件名和对应的标签存储在字典中
    d_meta = {}        #定义一个字典d_meta
    file_list=[]          #定义一个列表file_list
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()         #Python-with open() as f的用法,这里r是它的参数代表只读
                                        #读取指定文件（dir_meta）的内容，并将其存储在一个列表中（l_meta）

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
                                                          #逐行读取l_meta的内容，strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开
             file_list.append(key)               
             d_meta[key] = 1 if label == 'bonafide' else 0   #这里对于字典d_meta中: 键就是key,值是0或1
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]       #将x的行数赋值给x_len(输入应该是一个一维数组)
    if x_len >= max_len:   
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1   
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]  #np.tile()重复某个数组，这里x代表要被复制的数组，(1, num_repeats)代表复制的结果是1行num_repeats列的x数组
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo):   #继承了PyTorch的Dataset基类
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs   #一个音频文件的列表，存储每个音频文件的名称和路径
            self.labels = labels       #一个字典，将每个音频文件的名称映射为其相应的标签
            self.base_dir = base_dir   #音频文件的基础目录
            self.algo=algo             #需要使用的算法
            self.args=args             #一些其他参数的集合
            self.cut=64600 # take ~4 sec audio (64600 samples)         读取音频时取样64600个样本点（即约4秒音频）

	def __len__(self):          
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]               #通过self.list_IDs[index]获取指定索引位置的utt_id（音频文件的标识符）
            X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)     #X是采样数据，fs是采样率。采样数据代表了音频信号在不同时间点上的幅度。路径由self.base_dir（音频文件的基础目录）和utt_id组成，文件格式为FLAC，采样率为16000
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)    #使用process_Rawboost_feature方法(数据增强)对音频数据进行处理   
            X_pad= pad(Y,self.cut)        #调用pad方法进行填充
            x_inp= Tensor(X_pad)       
            target = self.labels[utt_id]
            
            return x_inp, target
            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id  



#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
