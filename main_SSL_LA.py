import argparse
import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import torch
torch.cuda.set_device(0)
from torch import nn                                                                                                                                                                        
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from model import SEGatedLinearConcatBottle2neck

import eval_metric_LA as em
import pandas 

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def evaluate_accuracy(dev_loader, model, device):    #评估模型在验证集上的准确性
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)  #权重向量weight包含两个类别的权重
    criterion = nn.CrossEntropyLoss(weight=weight)
# 加权交叉熵目标函数是一种用于训练模型的损失函数。
# 在分类任务中，我们希望模型能够正确地预测输入样本的类别。而加权交叉熵目标函数在计算损失时会对不同类别的样本赋予不同的权重。
# 加权交叉熵目标函数的计算方式是将模型的预测结果与真实标签进行比较，并通过对预测结果进行概率化，将其转换为各个类别的概率分布。
# 然后，根据真实标签的类别，选择相应的概率值作为损失函数的一部分。这样做的目的是对于不同的类别，模型会更关注那些被赋予较高权重的样本，从而更加有效地优化模型。
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)  #返回batch_x的第一个维度的大小，获取样本数量
        num_total += batch_size       #记录处理过的样本数量
        batch_x = batch_x.to(device)    
        batch_y = batch_y.view(-1).type(torch.int64).to(device)    #用于将 batch_y 转换为一维张量，保持元素总数不变
        
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)  #batch_loss是一个张量,它包含了一个批次中每个样本的损失
        val_loss += (batch_loss.item() * batch_size)  #这里把每一个样本的损失都*batch_size的目的是考虑每个样本在整个批次中的权重
        
    val_loss /= num_total
   
    return val_loss



def produce_evaluation_file(dataset, model, device, save_path): #生成评估文件
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)#drop_last=False是如果最后一个批次的样本数量小于指定的批次大小，则仍会保留该批次并进行处理
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]           #取第二列数据
                       ).data.cpu().numpy().ravel()    #抛到CPU，转成numpy，将多维数组展平为一维数组
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))




# def train_epoch(train_loader, model, lr,optim, device):
#     running_loss = 0
    
#     num_total = 0.0
    
#     model.train()

#     #set objective (Loss) functions
#     weight = torch.FloatTensor([0.1, 0.9]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)
    
#     for batch_x, batch_y in train_loader:
       
#         batch_size = batch_x.size(0)
#         num_total += batch_size
        
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.view(-1).type(torch.int64).to(device)
#         batch_out = model(batch_x)
        
#         batch_loss = criterion(batch_out, batch_y)
        
#         running_loss += (batch_loss.item() * batch_size)
       
#         optimizer.zero_grad()
#         batch_loss.backward()
#         optimizer.step()
       
#     running_loss /= num_total
    
#     return running_loss

asv_key_file = '/g813_u1/g813_u1/SSL_Anti-spoofing/LA-keys-stage-1/keys/ASV/trial_metadata.txt'
asv_scr_file = '/g813_u1/g813_u1/SSL_Anti-spoofing/LA-keys-stage-1/keys/ASV/ASVTorch_Kaldi/score.txt'
cm_key_file = '/g813_u1/g813_u1/SSL_Anti-spoofing/LA-keys-stage-1/keys/CM/trial_metadata.txt'


# if len(sys.argv) != 4:
#     print("CHECK: invalid input arguments. Please read the instruction below:")
#     print(__doc__)
#     exit(1)

# submit_file = sys[1]    #所生成的得分
# truth_dir = sys.argv[2]   #答案
phase = 'eval'

Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack 伪冒攻击的先验概率
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker 目标说话人的先验概率
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker 非目标说话人的先验概率
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker 拒绝目标说话人的成本
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker 接受非目标说话人的成本
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof 接受伪冒攻击的成本
}



def load_asv_metrics():

    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)
    
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


target_scores = np.array([])
nontarget_scores = np.array([])

def cat_scores(batch_score,batch_y,len):
    global target_scores
    global nontarget_scores
    for j in range(len):
        if(batch_y[j]==1):
            target_scores=np.append(target_scores,batch_score[j])
        else:
            nontarget_scores=np.append(nontarget_scores,batch_score[j])



def train_epoch(data_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train() # 作用是启用batch normalization和drop out
    # optim = torch.optim.Adam(model.parameters(), lr=lr) # 优化算法
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.NLLLoss(weight=weight)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in data_loader:

        batch_size = batch_x.size(0)# batch_size=32
        num_total += batch_size

        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().ravel()
        #二元分类模型的输出分数，第1列表示正类别的分数，第0列表示负类比的分数。差值表示模型对输入的正类别的置信度
 
        len = batch_y.shape[0]
        cat_scores(batch_score,batch_y,len)
        # cat_scores1(batch_score,batch_y)

        batch_loss = criterion(batch_out, batch_y)

        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

        running_loss += (batch_loss.item() * batch_size)
        # if ii % 10 == 0:
        #     # 输出正确率
        #     sys.stdout.write('\r \t {:.2f}aaaaaa'.format(
        #         (num_correct/num_total)*100))
            
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    
    train_accuracy = (num_correct/num_total)*100
# =============EER======================
    print()
    eer,threshold = em.compute_eer(target_scores,nontarget_scores)
    # eer,threshold = compute_eer(ground_true_label,ground_predict_scores)
    

    print("eer = ",eer)
    print("threshold = ",threshold)


# ===============tDCF=======================

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics()
    tDCF_curve , _ = em.compute_tDCF(target_scores, nontarget_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    print("min_tDCF = ",min_tDCF)

    return running_loss























if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='ASVspoof2019 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/g813_u1/mnt/2021/LA/', help='/g813_u1/mnt/2021/LA/')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    '''

    parser.add_argument('--protocols_path', type=str, default='/g813_u1/mnt/2021/', help='/g813_u1/mnt/2021/')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments辅助的参数
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path): 
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device,SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False,loss='oc-softmax')
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])  #param.view(-1)将张量 param 展平为一维张量，size()方法获取展平后张量的形状
    model =model.to(device)
    print('nb_params:',nb_params)  #获取参数总和

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    #学习率，用于控制每次参数更新的步长大小。学习率越大，参数更新的幅度越大，学习速度也会更快。
    #权重衰减参数，用于对模型的权重进行正则化，以防止过拟合。它控制了在参数更新中添加的额外衰减项的大小。较大的权重衰减值将对权重进行更强的惩罚
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device),strict=False)
        print('Model loaded : {}'.format(args.model_path))
    # model.load_state_dict()函数用于将加载的模型权重加载到当前模型对象中。
    # torch.load()函数加载模型权重文件，args.model_path是命令行参数中指定的模型路径。map_location=device参数用于指定将模型加载到哪个设备上。
    # strict=False参数表示允许加载模型权重时出现不匹配的键（比如预训练模型中的某些层在当前模型中不存在）  
    

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
        #genSpoof_lis生成用于评估的文件列表
        print('no. of eval trials',len(file_eval))#打印评估集中的样本数量
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)



    # define train dataloader
    d_label_trn,file_train = genSpoof_list(dir_meta = os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list(dir_meta = os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        
        val_loss = evaluate_accuracy(dev_loader, model, device)


        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
