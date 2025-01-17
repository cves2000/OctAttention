'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY SR
Description: attentionModel
FilePath: /compression/attentionModel.py
'''
import torch
import torch.nn as nn
import math
import copy
from networkTool import device

class SelfMultiheadAttention(nn.Module):
    def __init__(self, emsize, nhead, dropout=0.5):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # 头的数量
        self.head_size = emsize // nhead  # 每个头的大小
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  # 所有头的总大小
        self.mlpKey = nn.Linear(emsize, self.all_head_size) # 定义线性变换用于生成键
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)  # 定义线性变换用于生成查询
        self.mlpValue = nn.Linear(emsize, self.all_head_size)  # 定义线性变换用于生成值
        self.dropout = nn.Dropout(dropout)  # 定义dropout层

    def slice(self,x,dim):
        # 切片函数，用于将mlpKey、mlpQuery和mlpValue的输出切片以实现多头注意力
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size) # [batch_size, bptt, nhead, head_size] or [batch_size, bptt, levelNumK, nhead, head_size]
        x = x.view(*new_x_shape)
        if (dim == 3):
            x = x.permute(0, 2, 1, 3)
        elif (dim == 4):
            x = x.permute(0,1,3,2,4)
            assert 0
        return x

    def forward(self,em,mask):
        # 前向传播函数，计算多头注意力
        em = em.transpose(0,1).contiguous()  #[batch_size,bptt,...]
        Key = self.slice(self.mlpKey(em),em.dim())    # [batch_size, bptt, all_head_size] -> [batch_size,nhead,bptt,head_size]
        Query = self.slice(self.mlpQuery(em),em.dim()) #torch.Size([32, 4, 256, 42])
        Value = self.slice(self.mlpValue(em),em.dim())

        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)  #[batch_size,nhead,bptt,bptt] or [bs,bptt,nhead,levelNumK,levelNumK]
        attention_score = attention_score + mask #torch.Size([32, 4, 256, 256]) ,mask [[0,-inf,-inf,..],[0,0,-inf,...],[0,0,0,...]]
        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))

        context = torch.matmul(attention_map, Value)        # [batch_size, 4 nhead, bptt, 42 head_size] #torch.Size([32, 4, 256, 42])
        if (context.dim() == 4):
            context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, bptt, 4 nhead, 42 head_size]
        elif (context.dim()==5):
            context = context.permute(0, 1, 3, 2, 4).contiguous()  # [batch_size, bptt, levelNumK, 8, 64]
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0,1).contiguous()
        return context
 
class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.MultiAttention = SelfMultiheadAttention(emsize=ninp,nhead=nhead)  # 多头注意力层
        self.linear1 = nn.Linear(ninp,nhid)  # 线性变换层1
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = nn.Linear(nhid,ninp)  # 线性变换层2

        self.norm1 = nn.LayerNorm(ninp, eps=1e-5)  # 归一化层1
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)  # 归一化层2
        self.dropout1 = nn.Dropout(dropout)  # dropout层1
        self.dropout2 = nn.Dropout(dropout)  # dropout层2

    def forward(self, src, src_mask):
        # 前向传播函数，计算Transformer层的输出
        src2 = self.MultiAttention(src,src_mask)  #Multi-head Attention
        src = self.dropout1(src2) + src
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))  #[batch_size,bptt,ninp] -> [batch_size,bptt,nhid] -> [batch_size,bptt,ninp]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])  # Transformer层的列表

    def forward(self,src,src_mask):
        # 前向传播函数，计算Transformer模块的输出
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output
