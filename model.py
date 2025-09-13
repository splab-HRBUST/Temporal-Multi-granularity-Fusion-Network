import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

from torch.nn.init import kaiming_normal_, xavier_normal_
import math
from torch.nn import Parameter

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## FOR fine-tuned SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/g813_u1/mnt/g813_u1/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


#---------AASIST back-end------------------------#
''' Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans. 
    AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. 
    In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(nn.Module):       #实现了图注意力层，用于在图神经网络中对节点进行注意力加权聚合
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x
        
    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        #print('num_type1',num_type1)
        #print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        #print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        #print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('Concat x1 and x2',x.shape)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            #print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        #print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        #print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        #print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        #print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        #print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):
        
        x1 = self.proj_with_attM(torch.matmul(
           att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
   
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h




class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        #self.mp = nn.MaxPool2d((1, 2))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)
        return out





















class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()

    """

    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()

    def forward(self, inputs, target):
        """
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from OCAngle
                 inputs[0]: positive class score
                 inputs[1]: negative class score
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # Assume target is binary, positive (genuine) = 0, negative (spoofed) = 1
        #
        # Equivalent to select the scores using if-elese
        # if target = 1, use inputs[1]
        # else, use inputs[0]
        output = inputs[1] * target.view(-1, 1) + \
                 inputs[0] * (1 - target.view(-1, 1))
        loss = self.m_loss(output).mean()
        return loss
class OCAngleLayer(nn.Module):
    """ Output layer to produce activation for one-class softmax

    Usage example:
     batchsize = 64
     input_dim = 10
     class_num = 2

     l_layer = OCAngleLayer(input_dim)
     l_loss = OCSoftmaxWithLoss()

     data = torch.rand(batchsize, input_dim, requires_grad=True)
     target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
     target = target.to(torch.long)

     scores = l_layer(data)
     loss = l_loss(scores, target)

     loss.backward()
    """

    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1
        # print('inplanes:',self.in_planes)
        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        # self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations

        input:
        ------
          input tensor (batchsize, input_dim)

        output:
        -------
          tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        # print('input:',input)
        # print(input.shape)

        # print('self.weight:',self.weight.shape)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        # print('w:',w)
        # print(w.shape)

        x_modulus = input.pow(2).sum(1).pow(0.5)
        # print('x_modulus:',x_modulus)
        # print(x_modulus.shape)

        inner_wx = input.mm(w)
        # print('inner_wx:',inner_wx)
        # print(inner_wx.shape)

        cos_theta = inner_wx / x_modulus.view(-1, 1)
        # print('cos_theta',cos_theta)
        # print(cos_theta.shape)

        cos_theta = cos_theta.clamp(-1, 1)
        # print('cos_theta',cos_theta)
        # print(cos_theta.shape)

        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            # print('pos_score:',pos_score)
            # print(pos_score.shape)

            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
            # print('neg_score',neg_score)
            # print(neg_score.shape)

        out = torch.cat([neg_score, pos_score], dim=1)
        # print('out:',out)
        # print(out.shape)

        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print('se reduction: ', reduction)
        # print(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):  # x: B*C*D*T
        b, c, _, _ = x.size()
        #print("x = ",x.size())
        y = self.avg_pool(x).view(b, c)
        #print("avg : = ",y.size())
        y = self.fc(y).view(b, c, 1, 1)
        #print("y : = ",y.size())
        return x * y.expand_as(x)
class LinearConcatGate(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearConcatGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(indim, outdim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_prev, x):
        x_cat = torch.cat([x_prev, x], dim=1)
        b, c_double, _, _ = x_cat.size()
        c = int(c_double / 2)
        y = self.avg_pool(x_cat).view(b, c_double)
        y = self.sigmoid(self.linear(y)).view(b, c, 1, 1)
        return x_prev * y.expand_as(x_prev)





class SEGatedLinearConcatBottle2neck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal',
                 gate_reduction=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(SEGatedLinearConcatBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        if stype != 'stage':
            gates = []
            for i in range(self.nums - 1):
                gates.append(LinearConcatGate(2 * width, width))
            self.gates = nn.ModuleList(gates)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        # print('x: ', x.size())
        out = self.conv1(x)
        # print('conv1: ', out.size())
        out = self.bn1(out)
        out = self.relu(out)
        # print("out = ",out.shape)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = gate_sp + spx[i]
            sp = self.convs[i](sp)
            bn_sp = self.bns[i](sp)
            if self.stype != 'stage' and i < self.nums - 1:
                gate_sp = self.gates[i](sp, spx[i + 1])
            sp = self.relu(bn_sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # print('conv2: ', out.size())
        # print(self.width)
        # print(self.scale)
        out = self.conv3(out)
        # print('conv3: ', out.size())
        out = self.bn3(out)
        # print('bn3: ', out.size())
        out = self.se(out)
        # print('se :', out.size())

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out











class Model(nn.Module):
    def __init__(self, args,device, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000 ,loss='softmax', gate_reduction=4,
                 **kwargs):
        super().__init__()
        self.device=device
        
        
        filts =[128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims =[64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]




        self.inplanes = 16
        # super(GatedRes2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.gate_reduction = gate_reduction
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])  # 64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
      
        
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            print('block',block.expansion)
            self.cls_layer = nn.Sequential(nn.Linear(128 * block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()
        elif self.loss == 'oc-softmax':
            #print('block',block.expansion)
            self.cls_layer = OCAngleLayer(128 * block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)







        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        
        self.conv = nn.Conv2d(1,16,(1,1))
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.first_bn = nn.BatchNorm2d(num_features=1)  #num_features指输入特征图的通道数
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)   #随机丢弃输入的一部分特征
        self.selu = nn.SELU(inplace=True)


        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)







    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale,
                  gate_reduction=self.gate_reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale,
                      gate_reduction=self.gate_reduction))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) # (bs,frame_number,feat_out_dim) 
                              # bs代表批量大小；
                              # frame_number代表帧数或时间步数，表示时间序列数据中的帧的数量；
                              # feat_out_dim表示特征输出维度，表示每个帧或时间步的特征向量的维度
     

        # post-processing on front-end features
        x= x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)  (bs，128, 201)
        x = x.unsqueeze(dim=1) # add channel dim=1 是指定要在第1个维度（索引从0开始）上插入一个新的维度
        x = F.max_pool2d(x, (3, 3))   #(1,42,67)
        x = self.first_bn(x)
        x = self.selu(x)

        # #RawNet2-based encoder
        # x = self.encoder(x)       
        # x = self.first_bn1(x)
        # x = self.selu(x)
        
        #CNN
      
        x11 = x

        x21 = x[:, :, :, :33]
        x22 = x[:, :, :, 33:]

        x31 = x[:, :, :, 0:16]
        x32 = x[:, :, :, 16:32]
        x33 = x[:, :, :, 32:48]
        x34 = x[:, :, :, 48:64]
        
        x41 = x[:, :, :, 0:8]
        x42 = x[:, :, :, 8:16]
        x43 = x[:, :, :, 16:24]
        x44 = x[:, :, :, 24:32]
        x45 = x[:, :, :, 32:40]
        x46 = x[:, :, :, 40:48]
        x47 = x[:, :, :, 48:56]
        x48 = x[:, :, :, 56:64]

        out11 = self.pool(torch.relu(self.conv(x11)))

        out21 = self.pool(torch.relu(self.conv(x21)))
        
        out22 = self.pool(torch.relu(self.conv(x22)))     
        
        out31 = self.pool(torch.relu(self.conv(x31)))
        out32 = self.pool(torch.relu(self.conv(x32)))
        out33 = self.pool(torch.relu(self.conv(x33)))
        out34 = self.pool(torch.relu(self.conv(x34)))

        out41 = self.pool(torch.relu(self.conv(x41)))
        out42 = self.pool(torch.relu(self.conv(x42)))
        out43 = self.pool(torch.relu(self.conv(x43)))
        out44 = self.pool(torch.relu(self.conv(x44)))
        out45 = self.pool(torch.relu(self.conv(x45)))
        out46 = self.pool(torch.relu(self.conv(x46)))
        out47 = self.pool(torch.relu(self.conv(x47)))
        out48 = self.pool(torch.relu(self.conv(x48)))

        out = torch.cat((out11,out21,out22,out31,out32,out33,out34,out41,out42,out43,out44,out45,out46,out47,out48),dim=3)



        x = self.layer1(out)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)
        # print('cls_layer: ', x.size())
        return x













        # w=self.attention(out)
        
        # #------------SAP for spectral feature-------------对声谱特征进行注意力加权池化#
        # w1=F.softmax(w,dim=-1)       #(10,64,42,67)
        # m = torch.sum(out * w1, dim=-1)   #(10,64,42)      
        # e_S = m.transpose(1, 2) + self.pos_S  #(10,42,64)
        

        # # graph module layer
        # gat_S = self.GAT_layer_S(e_S)        
        # out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)(10,21,64)
        

        
        # #------------SAP for temporal feature-------------针对时间特征的自注意力池化#
        # w2=F.softmax(w,dim=-2)       #(10,64,42,67)
        # m1 = torch.sum(out * w2, dim=-2)  #(10,64,67)    
        # e_T = m1.transpose(1, 2)
       

        # # graph module layer
        # gat_T = self.GAT_layer_T(e_T)
        # out_T = self.pool_T(gat_T)
        
        # # learnable master node
        # master1 = self.master1.expand(x.size(0), -1, -1)
        # master2 = self.master2.expand(x.size(0), -1, -1)

        # # inference 1
        # out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
        #     out_T, out_S, master=self.master1)

        # out_S1 = self.pool_hS1(out_S1)
        # out_T1 = self.pool_hT1(out_T1)

        # out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
        #     out_T1, out_S1, master=master1)
        # out_T1 = out_T1 + out_T_aug
        # out_S1 = out_S1 + out_S_aug
        # master1 = master1 + master_aug

        # # inference 2
        # out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
        #     out_T, out_S, master=self.master2)
        # out_S2 = self.pool_hS2(out_S2)
        # out_T2 = self.pool_hT2(out_T2)

        # out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
        #     out_T2, out_S2, master=master2)
        # out_T2 = out_T2 + out_T_aug
        # out_S2 = out_S2 + out_S_aug
        # master2 = master2 + master_aug

        # out_T1 = self.drop_way(out_T1)
        # out_T2 = self.drop_way(out_T2)
        # out_S1 = self.drop_way(out_S1)
        # out_S2 = self.drop_way(out_S2)
        # master1 = self.drop_way(master1)
        # master2 = self.drop_way(master2)

        # out_T = torch.max(out_T1, out_T2)
        # out_S = torch.max(out_S1, out_S2)
        # master = torch.max(master1, master2)

        # T_max, _ = torch.max(torch.abs(out_T), dim=1)
        # T_avg = torch.mean(out_T, dim=1)

        # S_max, _ = torch.max(torch.abs(out_S), dim=1)
        # S_avg = torch.mean(out_S, dim=1)

        # last_hidden = torch.cat(
        #     [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        # last_hidden = self.drop(last_hidden)
        
        # output = self.out_layer(last_hidden)
        
        # return output
