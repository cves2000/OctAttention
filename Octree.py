'''
作者: fuchy@stu.pku.edu.cn
描述: 八叉树 
文件路径: /compression/Octree.py
版权所有。
'''
import numpy as np
from OctreeCPP.Octreewarpper import GenOctree

# 定义八叉树节点类
class CNode():
    def __init__(self,nodeid=0,childPoint=[[]]*8,parent=0,oct=0,pos = np.array([0,0,0]),octant = 0) -> None:
        self.nodeid = nodeid  # 节点ID
        self.childPoint=childPoint.copy()  # 子节点
        self.parent = parent  # 父节点
        self.oct = oct  # 占用代码，范围1~255
        self.pos = pos  # 位置
        self.octant = octant  # 八分体，范围1~8

# 定义八叉树类
class COctree():
    def __init__(self,node=[],level=0) -> None:
        self.node = node.copy()  # 节点
        self.level=level  # 层级

# 将十进制数转换为二进制数，使用count位数
def dec2bin(n, count=8): 
    """returns the binary of integer n, using count number of digits""" 
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]

# 将十进制数组转换为二进制数组
def dec2binAry(x, bits):
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1).T              
    return (np.bitwise_and(np.expand_dims(x,1), mask)!=0).astype(int) 

# 将二进制数组转换为十进制数
def bin2decAry(x):
    if(x.ndim==1):
        x = np.expand_dims(x,0)
    bits = x.shape[1]
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1)
    return x.dot(mask).astype(int)

# 实现Morton编码，将多维空间索引转换为一维空间索引
def Morton(A):
    A =  A.astype(int)
    n = np.ceil(np.log2(np.max(A)+1)).astype(int)   
    x = dec2binAry(A[:,0],n)                         
    y = dec2binAry(A[:,1],n)
    z = dec2binAry(A[:,2],n)
    m = np.stack((x,y,z),2)                           
    m = np.transpose(m,(0, 2, 1))                     
    mcode = np.reshape(m,(A.shape[0],3*n),order='F')  
    return mcode
 
# 实现八叉树的解码，将Morton编码转换回原始的点云数据
def DeOctree(Codes):
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes,8),axis=1)  
    codeL = occupancyCode.shape[0]                        
    N = np.ones((30),int) 
    codcal = 0
    L = 0
    while codcal+N[L]<=codeL:
        L +=1
        try:
            N[L+1] = np.sum(occupancyCode[codcal:codcal+N[L],:])
        except:
            assert 0
        codcal = codcal+N[L]
    Lmax = L
    Octree = [COctree() for _ in range(Lmax+1)]
    proot = [np.array([0,0,0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1,Lmax+1):
        childNode = []  # 下一级的节点
        for currentNode in Octree[L-1].node: # 当前节点的bbox
            code = occupancyCode[codei,:]
            for bit in np.where(code==1)[0].tolist():
                newnode =currentNode+(np.array(dec2bin(bit, count=3))<<(Lmax-L))# 子节点的bbox
                childNode.append(newnode)
            codei+=1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points

# 生成K级父序列
def GenKparentSeq(Octree,K):
    LevelNum =len(Octree)
    nodeNum = Octree[-1].node[-1].nodeid
    Seq = np.ones((nodeNum,K),'int')*255
    LevelOctant = np.zeros((nodeNum,K,2),'int') # 层级和八分体
    Pos = np.zeros((nodeNum,K,3),'int'); # 填充0
    ChildID = [[] for _ in range(nodeNum)]
    Seq[0,K-1] = Octree[0].node[0].oct
    Seq[0,0:K-1] = Seq[node.parent-1,1:K]
    LevelOctant[0,K-1,0] = 1
    LevelOctant[0,K-1,1] = 1
    Pos[0,K-1,:] = Octree[0].node[0].pos
    Octree[0].node[0].parent = 1 # 设置为1
    n= 0 
    for L in range(0,LevelNum):
        for node in Octree[L].node:
            Seq[n,K-1] = node.oct
            Seq[n,0:K-1] = Seq[node.parent-1,1:K]
            LevelOctant[n,K-1,:] = [L+1,node.octant]
            LevelOctant[n,0:K-1] = LevelOctant[node.parent-1,1:K,:]
            Pos[n,K-1] = node.pos
            Pos[n,0:K-1,:] = Pos[node.parent-1,1:K,:]
            if (L==LevelNum-1):
                pass
            n+=1
    assert n==nodeNum
    DataStruct = {'Seq':Seq,'Level':LevelOctant,'ChildID':ChildID,'Pos':Pos}
    return DataStruct
