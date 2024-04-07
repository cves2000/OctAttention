'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''
#%%
import  numpy as np
import torch
from tqdm import tqdm
from Octree import DeOctree, dec2bin
import pt 
from dataset import default_loader as matloader
from collections import deque
import os 
import time
from networkTool import *
from encoderTool import generate_square_subsequent_mask
from encoder import model,list_orifile
import numpyAc
batch_size = 1 
bpttRepeatTime = 1
#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile,oct_data_seq,model,bptt):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        elapsed = time.time()  # 记录开始时间

        KfatherNode = [[255,0,0]]*levelNumK  # 初始化KfatherNode
        nodeQ = deque()  # 创建一个双端队列
        oct_seq = []  # 初始化oct_seq
        src_mask = generate_square_subsequent_mask(bptt).to(device)  # 生成源掩码

        input = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)  # 初始化输入
        padinginbptt = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)  # 初始化padinginbptt
        bpttMovSize = bptt//bpttRepeatTime  # 计算bpttMovSize

        output = model(input,src_mask,[])  # 计算模型的输出

        freqsinit = torch.softmax(output[-1],1).squeeze().cpu().detach().numpy()  # 计算freqsinit
        
        oct_len = len(oct_data_seq)  # 计算oct_len

        dec = numpyAc.arithmeticDeCoding(None,oct_len,255,binfile)  # 进行算术解码

        root =  decodeNode(freqsinit,dec)  # 解码节点
        nodeId = 0  # 初始化nodeId
        
        KfatherNode = KfatherNode[3:]+[[root,1,1]] + [[root,1,1]]  # 更新KfatherNode
        
        nodeQ.append(KfatherNode)  # 将KfatherNode添加到nodeQ
        oct_seq.append(root)  # 将root添加到oct_seq  
        
        with tqdm(total=  oct_len+10) as pbar:  # 创建一个进度条
            while True:  # 开始循环
                father = nodeQ.popleft()  # 从nodeQ中弹出一个元素
                childOcu = dec2bin(father[-1][0])  # 计算childOcu
                childOcu.reverse()  # 反转childOcu
                faterLevel = father[-1][1]  # 计算faterLevel
                for i in range(8):  # 遍历范围
                    if(childOcu[i]):  # 如果childOcu[i]为真
                        faterFeat = [[father+[[root,faterLevel+1,i+1]]]]  # 计算faterFeat
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)  # 将faterFeat转换为张量并移动到设备上
                        faterFeatTensor[:,:,:,0] -= 1  # 更新faterFeatTensor

                        # shift bptt window
                        offsetInbpttt = (nodeId)%(bpttMovSize)  # 计算offsetInbpttt
                        if offsetInbpttt==0:  # 如果offsetInbpttt为0
                            input = torch.vstack((input[bpttMovSize:],faterFeatTensor,padinginbptt[0:bpttMovSize-1]))  # 更新输入
                        else:
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor  # 更新输入

                        output = model(input,src_mask,[])  # 计算模型的输出
                        
                        Pro = torch.softmax(output[offsetInbpttt+bptt-bpttMovSize],1).squeeze().cpu().detach().numpy()  # 计算Pro

                        root =  decodeNode(Pro,dec)  # 解码节点
                        nodeId += 1  # 更新nodeId
                        pbar.update(1)  # 更新进度条
                        KfatherNode = father[1:]+[[root,faterLevel+1,i+1]]  # 更新KfatherNode
                        nodeQ.append(KfatherNode)  # 将KfatherNode添加到nodeQ
                        if(root==256 or nodeId==oct_len):  # 如果root等于256或nodeId等于oct_len
                            assert len(oct_data_seq) == nodeId  # 检查oct的数量
                            Code = oct_seq  # 获取Code
                            return Code,time.time() - elapsed  # 返回Code和耗时
                        oct_seq.append(root)  # 将root添加到oct_seq
                    assert oct_data_seq[nodeId] == root  # 检查

def decodeNode(pro,dec):
    root = dec.decode(np.expand_dims(pro,0))  # 解码节点
    return root+1  # 返回解码后的节点


if __name__=="__main__":

    for oriFile in list_orifile:  # 从encoder.py中获取
        ptName = os.path.basename(oriFile)[:-4]  # 获取文件名（不包括扩展名）
        matName = 'Data/testPly/'+ptName+'.mat'  # 获取mat文件名
        binfile = expName+'/data/'+ptName+'.bin'  # 获取bin文件名
        cell,mat =matloader(matName)  # 加载mat文件

        # 读取Sideinfo
        oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-1:,0]  # 用于检查
        
        p = np.transpose(mat[cell[1,0]]['Location'])  # 原始点云
        offset = np.transpose(mat[cell[2,0]]['offset'])  # 偏移量
        qs = mat[cell[2,0]]['qs'][0]  # qs

        Code,elapsed = decodeOct(binfile,oct_data_seq,model,bptt)  # 解码Oct
        print('decode succee,time:', elapsed)  # 打印"解码成功，时间："
        print('oct len:',len(Code))  # 打印"oct长度："

        # DeOctree
        ptrec = DeOctree(Code)  # DeOctree
        # Dequantization
        DQpt = (ptrec*qs+offset)  # Dequantization
        pt.write_ply_data(expName+"/temp/test/rec.ply",DQpt)  # 写入ply数据
        pt.pcerror(p,DQpt,None,'-r 1',None).wait()
