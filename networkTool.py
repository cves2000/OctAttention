'''
作者: fuchy@stu.pku.edu.cn
最后编辑者: 请设置最后编辑者
描述: 网络参数和辅助函数
文件路径: /compression/networkTool.py
'''

import torch  # 导入torch库
import os,random  # 导入os和random库
import numpy as np  # 导入numpy库
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置环境变量，指定CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果CUDA可用，设备设置为CUDA，否则设置为CPU

# 网络参数
bptt = 1024  # 上下文窗口长度
expName = './Exp/Obj'  # 实验名称
DataRoot = './Data/Obj'  # 数据根目录

checkpointPath = expName+'/checkpoint'  # 检查点路径
levelNumK = 4  # 等级数

trainDataRoot = DataRoot+"/train/*.mat"  # 训练数据根目录
expComment = 'OctAttention, 在MPEG 8i,MVUB 1~10级上训练。2021/12. 版权所有。'  # 实验注释

MAX_OCTREE_LEVEL = 12  # 八叉树最大等级
# 随机种子
seed=2
torch.manual_seed(seed)  # 设置torch的随机种子
torch.cuda.manual_seed(seed)  # 设置CUDA的随机种子
torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机种子
np.random.seed(seed)  # 设置numpy的随机种子
random.seed(seed)  # 设置random的随机种子
torch.backends.cudnn.benchmark=False  # 关闭cudnn的基准测试
torch.backends.cudnn.deterministic=True  # 开启cudnn的确定性模式
os.environ["H5PY_DEFAULT_READONLY"] = "1"  # 设置环境变量，使h5py默认为只读模式

# 工具函数
def save(index, saveDict,modelDir='checkpoint',pthType='epoch'):  # 保存函数
    if os.path.dirname(modelDir)!='' and not os.path.exists(os.path.dirname(modelDir)):  # 如果模型目录不存在
        os.makedirs(os.path.dirname(modelDir))  # 创建模型目录
    torch.save(saveDict, modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, index))  # 保存模型

def reload(checkpoint,modelDir='checkpoint',pthType='epoch',print=print,multiGPU=False):  # 重新加载函数
    try:
        if checkpoint is not None:  # 如果检查点不为空
            saveDict = torch.load(modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint),map_location=device)  # 加载模型
            pth = modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)  # 获取模型路径
        if checkpoint is None:  # 如果检查点为空
            saveDict = torch.load(modelDir,map_location=device)  # 加载模型
            pth = modelDir  # 获取模型路径
        saveDict['path'] = pth  # 将模型路径添加到saveDict中
        if multiGPU:  # 如果使用多GPU
            from collections import OrderedDict  # 导入有序字典
            state_dict = OrderedDict()  # 创建一个有序字典
            new_state_dict = OrderedDict()  # 创建一个有序字典
            for k, v in saveDict['encoder'].items():  # 遍历saveDict中的'encoder'项
                name = k[7:]  # 移除`module.`
                state_dict[name] = v  # 将v添加到state_dict中
            saveDict['encoder'] = state_dict  # 将state_dict添加到saveDict中
        return saveDict  # 返回saveDict
    except Exception as e:  # 捕获异常
        print('**warning**',e,' start from initial model')  # 打印警告信息
    return None  # 返回None

class CPrintl():  # 定义CPrintl类
    def __init__(self,logName) -> None:  # 初始化函数
        self.log_file = logName  # 设置日志文件名
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):  # 如果日志目录不存在
            os.makedirs(os.path.dirname(logName))  # 创建日志目录
    def __call__(self, *args):  # 调用函数
        print(*args)  # 打印参数
        print(*args, file=open(self.log_file, 'a'))  # 将参数写入日志文件

def model_structure(model,print=print):  # 定义model_structure函数
    print('-'*120)  # 打印分隔符
    print('|'+' '*30+'weight name'+' '*31+'|' \
            +' '*10+'weight shape'+' '*10+'|' \
            +' '*3+'number'+' '*3+'|')  # 打印表头
    print('-'*120)  # 打印分隔符
    num_para = 0  # 初始化参数数量
    for _, (key, w_variable) in enumerate(model.named_parameters()):  # 遍历模型的命名参数
        each_para = 1  # 初始化每个参数的数量
        for k in w_variable.shape:  # 遍历w_variable的形状
            each_para *= k  # 计算每个参数的数量
        num_para += each_para  # 更新参数数量

        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape), each_para))  # 打印参数信息
    print('-'*120)  # 打印分隔符
    print('The total number of parameters: ' + str(num_para))  # 打印总参数数量
    print('-'*120)  # 打印分隔符
