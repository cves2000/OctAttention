'''
作者: fuchy@stu.pku.edu.cn
最后编辑: 请设置最后编辑
描述: 
'''
from ctypes import *  # 导入ctypes模块，用于提供C兼容的数据类型
import numpy as np  # 导入numpy模块，用于数组和矩阵运算
import os  # 导入os模块，用于处理文件和目录
import numpy.ctypeslib as npct  # 导入numpy.ctypeslib模块，用于提供numpy和ctypes之间的互操作性

# 定义Node类，继承自ctypes的Structure类
class Node(Structure):
    _fields_ = [  # 定义结构体的字段
        ('nodeid',c_uint),  # 节点ID，类型为无符号整型
        ('octant',c_uint),  # 八分体，类型为无符号整型
        ('parent',c_uint),  # 父节点，类型为无符号整型
        ('oct',c_uint8),  # 八叉树，类型为8位无符号整型
        # ('pointIdx',c_void_p),
        ('pos',c_uint*3)  # 位置，类型为无符号整型数组，长度为3
    ]

c_double_p = POINTER(c_double)  # 定义c_double_p为指向c_double的指针
c_uint16_p = POINTER(c_uint16)  # 定义c_uint16_p为指向c_uint16的指针

lib = cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__))+'/Octree_python_lib.so')  # 加载动态链接库

lib.new_vector.restype = c_void_p  # 设置new_vector函数的返回类型为void指针
lib.new_vector.argtypes = []  # 设置new_vector函数的参数类型为空

lib.delete_vector.restype = None  # 设置delete_vector函数的返回类型为None
lib.delete_vector.argtypes = [c_void_p]  # 设置delete_vector函数的参数类型为void指针

lib.vector_size.restype = c_int  # 设置vector_size函数的返回类型为整型
lib.vector_size.argtypes = [c_void_p]  # 设置vector_size函数的参数类型为void指针

lib.vector_get.restype = c_void_p  # 设置vector_get函数的返回类型为void指针
lib.vector_get.argtypes = [c_void_p, c_int]  # 设置vector_get函数的参数类型为void指针和整型

lib.vector_push_back.restype = None  # 设置vector_push_back函数的返回类型为None
lib.vector_push_back.argtypes = [c_void_p, c_int]  # 设置vector_push_back函数的参数类型为void指针和整型

lib.genOctreeInterface.restype = c_void_p  # 设置genOctreeInterface函数的返回类型为void指针
lib.genOctreeInterface.argtypes = [c_void_p ,c_double_p,c_int]  # 设置genOctreeInterface函数的参数类型为void指针，double指针和整型

lib.Nodes_get.argtypes = [c_void_p,c_int]  # 设置Nodes_get函数的参数类型为void指针和整型
lib.Nodes_get.restype = POINTER(Node)  # 设置Nodes_get函数的返回类型为Node指针

lib.Nodes_size.restype = c_int  # 设置Nodes_size函数的返回类型为整型
lib.Nodes_size.argtypes = [c_void_p]  # 设置Nodes_size函数的参数类型为void指针

lib.int_size.restype = c_int  # 设置int_size函数的返回类型为整型
lib.int_size.argtypes = [c_void_p]  # 设置int_size函数的参数类型为void指针

lib.int_get.restype = c_int  # 设置int_get函数的返回类型为整型
lib.int_get.argtypes = [c_void_p,c_int]  # 设置int_get函数的参数类型为void指针和整型

# 定义COctree类
class COctree(object):

    def __init__(self):  # 初始化函数
        self.vector = lib.new_vector()  # 调用new_vector函数，返回值赋给self.vector
        self.code = None  # 初始化self.code为None

    def __del__(self):  # 析构函数
        lib.delete_vector(self.vector)  # 调用delete_vector函数，删除self.vector

    def __len__(self):  # 定义长度函数，返回vector的大小
        return lib.vector_size(self.vector)

    def __getitem__(self, i):  # 定义获取元素函数，返回vector的第i个元素
        L = self.__len__()
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return Level(lib.vector_get(self.vector, c_int(i)),i)
        
    def __repr__(self):  # 定义打印函数，打印vector的所有元素
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def push(self, i):  # 定义push函数，向vector中添加元素
        lib.vector_push_back(self.vector, c_int(i))

    def genOctree(self, p):  # 定义genOctree函数，生成八叉树
        data = np.ascontiguousarray(p).astype(np.double)
        data_p = data.ctypes.data_as(c_double_p)
        self.code = OctCode(lib.genOctreeInterface(self.vector,data_p,data.shape[0]))

# 定义OctCode类
class OctCode():
    def __init__(self,Adr) -> None:  # 初始化函数
        self.nodeAdr = Adr
        self.Len = lib.int_size(Adr)

    def __repr__(self):  # 定义打印函数，打印所有元素
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def __getitem__(self, i):  # 定义获取元素函数，返回第i个元素
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.int_get(self.nodeAdr,i)

    def __len__(self):  # 定义长度函数，返回元素个数
        return lib.int_size(self.nodeAdr)

# 定义Level类
class Level():
    def __init__(self, Adr,i) -> None:  # 初始化函数
        self.Adr = Adr
        self.node = Node(Adr)
        self.level = i+1
        self.Len = lib.Nodes_size(self.Adr)

    def __getitem__(self, i):  # 定义获取元素函数，返回第i个元素
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.Nodes_get(self.Adr,i).contents

    def __len__(self):  # 定义长度函数，返回元素个数
        return lib.Nodes_size(self.Adr)

# 定义Node类
class Node():
    def __init__(self,Adr) -> None:  # 初始化函数
        self.nodeAdr = Adr
        self.Len = lib.Nodes_size(Adr)

    def __getitem__(self, i):  # 定义获取元素函数，返回第i个元素
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.Nodes_get(self.nodeAdr,i).contents

    def __len__(self):  # 定义长度函数，返回元素个数
        return lib.Nodes_size(self.nodeAdr)

# 定义GenOctree函数，生成八叉树
def GenOctree(points):
    Octree = COctree()
    Octree.genOctree(points)
    return list(Octree.code),Octree,len(Octree)
