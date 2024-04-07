'''
作者: fuchy@stu.pku.edu.cn
描述: 这个文件用于编码点云
文件路径: /compression/encoder.py
版权所有。
'''
from numpy import mod  # 导入numpy模块的mod函数
from Preparedata.data import dataPrepare  # 导入数据准备模块
from encoderTool import main  # 导入编码工具
from networkTool import reload,CPrintl,expName,device  # 导入网络工具
from octAttention import model  # 导入模型
import glob,datetime,os  # 导入glob, datetime, os模块
import pt as pointCloud  # 导入点云处理模块

# 警告：decoder.py依赖于此处的模型，不要将此行移动到其他地方
model = model.to(device)  # 将模型加载到设备上
saveDic = reload(None,'modelsave/obj/encoder_epoch_00800093.pth')  # 重新加载模型
model.load_state_dict(saveDic['encoder'])  # 加载模型的状态字典

# 定义要处理的文件列表
list_orifile = ['file/Ply/2851.ply']
if __name__=="__main__":
    printl = CPrintl(expName+'/encoderPLY.txt')  # 创建日志打印对象
    printl('_'*50,'OctAttention V0.4','_'*50)  # 打印日志
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))  # 打印当前时间
    printl('load checkpoint', saveDic['path'])  # 打印加载的检查点路径
    for oriFile in list_orifile:  # 遍历要处理的文件列表
        printl(oriFile)  # 打印当前处理的文件名
        if (os.path.getsize(oriFile)>300*(1024**2)):  # 如果文件大小超过300M
            printl('too large!')  # 打印"too large!"
            continue  # 跳过当前文件，处理下一个文件
        ptName = os.path.splitext(os.path.basename(oriFile))[0]  # 获取文件名（不包括扩展名）
        for qs in [1]:  # 遍历qs列表
            ptNamePrefix = ptName  # 设置ptNamePrefix为ptName
            matFile,DQpt,refPt = dataPrepare(oriFile,saveMatDir='./Data/testPly',qs=qs,ptNamePrefix='',rotation=False)  # 准备数据
            # 当处理MVUB数据时，请在`dataPrepare`函数中设置`rotation=True`
            main(matFile,model,actualcode=True,printl =printl)  # 调用main函数进行编码，actualcode=False: 不会生成bin文件
            print('_'*50,'pc_error','_'*50)  # 打印'pc_error'
            pointCloud.pcerror(refPt,DQpt,None,'-r 1023',None).wait()  # 计算并打印点云误差
