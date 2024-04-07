import os
import os.path
import numpy as np
import glob
import torch.utils.data as data
from PIL import Image
import glob
# import scipy.io as scio
import h5py
from networkTool import trainDataRoot,levelNumK
IMG_EXTENSIONS = [
    'MPEG',
    'MVUB'
]

# 检查文件名是否为图像文件
def is_image_file(filename):
    return any(extension in filename for extension in IMG_EXTENSIONS)

# 默认的数据加载器
def default_loader(path):
    mat = h5py.File(path)
    # data = scio.loadmat(path)
    cell = mat['patchFile']
    return cell,mat

class DataFolder(data.Dataset):
    """ ImageFolder 可以用来加载没有标签的图像。"""

    def __init__(self, root, TreePoint,dataLenPerFile, transform=None ,loader=default_loader): 
         
        # dataLenPerFile 是一个 'mat' 文件中所有八叉树节点的平均数量
        
        dataNames = []
        for filename in sorted(glob.glob(root)):
            if is_image_file(filename):
                dataNames.append('{}'.format(filename))
        self.root = root
        self.dataNames =sorted(dataNames)
        self.transform = transform
        self.loader = loader
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        self.TreePoint = TreePoint
        self.fileLen = len(self.dataNames)
        assert self.fileLen>0,'no file found!'
        self.dataLenPerFile = dataLenPerFile # 你可以用 'calcdataLenPerFile' 中的确定的数字替换 'dataLenPerFile'
        # self.dataLenPerFile = self.calcdataLenPerFile() # 在你运行了 'calcdataLenPerFile' 之后，你可以注释掉这一行
        
    # 计算每个文件的数据长度
    def calcdataLenPerFile(self):
        dataLenPerFile = 0
        for filename in self.dataNames:
            cell,mat = self.loader(filename)
            for i in range(cell.shape[1]):
                dataLenPerFile+= mat[cell[0,i]].shape[2]
        dataLenPerFile = dataLenPerFile/self.fileLen
        print('dataLenPerFile:',dataLenPerFile,'you just use this function for the first time')
        return dataLenPerFile

    # 获取项目
    def __getitem__(self, index):
        while(self.index+self.TreePoint>self.datalen):
            filename = self.dataNames[self.fileIndx]
            # print(filename)
            if self.dataBuffer:
                a = [self.dataBuffer[0][self.index:].copy()]
            else:
                a=[]
            cell,mat = self.loader(filename)
            for i in range(cell.shape[1]):
                data = np.transpose(mat[cell[0,i]]) #shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
                data[:,:,0] = data[:,:,0] - 1
                a.append(data[:,-levelNumK:,:])# only take levelNumK level feats
                
            self.dataBuffer = []
            self.dataBuffer.append(np.vstack(tuple(a)))

            self.datalen = self.dataBuffer[0].shape[0]
            self.fileIndx+=200  # shuffle step = 1, will load continuous mat
            self.index = 0
            if(self.fileIndx>=self.fileLen):
                self.fileIndx=index%self.fileLen
        # print(index)  
        # try read
        img = []
        img.append(self.dataBuffer[0][self.index:self.index+self.TreePoint])

        self.index+=self.TreePoint

        if self.transform is not None:
            img = self.transform(img)
        return img

    # 获取长度
    def __len__(self):
        return int(self.dataLenPerFile*self.fileLen/self.TreePoint) # dataLen = octlen in total/TreePoint

        
if __name__=="__main__":

    TreePoint = 4096*16 # 数据中连续占用代码的数量，TreePoint*batch_size 可以被 batchSize 整除
    batchSize = 32
    train_set = DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile=356484.1) # 将加载 (batch_size,TreePoint,...) 形状的数据
    train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
    print('总八叉树(TreePoint*7): {}; 总批次: {}'.format(len(train_set), len(train_loader)))

    for batch, d in enumerate(train_loader):
        data_source = d[0].reshape((batchSize,-1,4,6)).permute(1,0,2,3) #d[0] 用于几何，d[1] 用于属性
        print(batch,data_source.shape)
        # print(data_source[:,0,:,0])
        # print(d[0][0],d[0].shape)
# %%
