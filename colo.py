import torch
import  torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

class LowLevelNet(nn.Module):
    def __init__(self):
        super(LowLevelNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,128,3,1,1)
        self.conv3=nn.Conv2d(128,128,3,2,1)
        self.conv4=nn.Conv2d(128,256,3,1,1)
        self.conv5=nn.Conv2d(256,256,3,2,1)
        self.conv6=nn.Conv2d(256,512,3,1,1)

    def forward(self,x):
        out = nn.ReLU(self.conv1(x))
        out = nn.ReLU(self.conv2(out))
        out = nn.ReLU(self.conv3(out))
        out = nn.ReLU(self.conv4(out))
        out = nn.ReLU(self.conv5(out))
        out = nn.ReLU(self.conv6(out))
        return out

class MidLevelNet(nn.Module):
    def __init__(self):
        super(MidLevelNet, self).__init__()
        self.conv1=nn.Conv2d(512,512,3,1,1)
        self.conv2=nn.Conv2d(512,256,3,1,1)

    def forward(self,x):
        out = nn.ReLU(self.conv1(x))
        out = nn.ReLU(self.conv2(out))
        return out

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__()
        self.conv1=nn.Conv2d(512,512,3,2,1)
        self.conv2=nn.Conv2d(512,512,3,1,1),
        self.conv3=nn.Conv2d(512,512,3,2,1),
        self.conv4=nn.Conv2d(512,512,3,1,1),
        self.fc1=nn.Linear(7*7*512,1024),#如果按照论文的数据集7*7*
        self.fc2=nn.Linear(1024,512),
        self.fc3=nn.Linear(512,256),

    def forward(self,x):
        out = nn.ReLU(self.conv1(x))
        out = nn.ReLU(self.conv2(out))
        out = nn.ReLU(self.conv3(out))
        out = nn.ReLU(self.conv4(out))
        out = out.view(-1,7*7*512)  #如果按照论文的数据集7*7*
        out = nn.ReLU(self.fc1(out))
        out = nn.ReLU(self.fc2(out))

        classIn = out

        out = nn.ReLU(self.fc3(out))
        fusionIn = out
        return fusionIn,classIn

class ClassNet(nn.Module):
    def __init__(self,numClasses):
        super(ClassNet, self).__init__()
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,numClasses)

    def forward(self,x):
        out=nn.ReLU(self.fc1(x))
        out=self.fc2(out)
        return out


class ColorizeNet(nn.Module):
    def __init__(self):
        super(ColorizeNet, self).__init__()
        self.conv1=nn.Conv2d(256,128,3,1,1)
        self.conv2=nn.Conv2d(128,64,3,1,1)
        self.conv3=nn.Conv2d(64,64,3,1,1)
        self.conv4=nn.Conv2d(64,32,3,1,1)
        self.conv5=nn.Conv2d(32,2,3,1,1)

    def forward(self,x):
        out=nn.ReLU(self.conv1(x))

        out=nn.functional.interpolate(out,scale_factor=2)
        out=nn.ReLU(self.con2(out))
        out=nn.ReLU(self.conv3(out))

        out=nn.functional.interpolate(out,scale_factor=2)
        out=nn.ReLU(self.conv4(out))
        out=nn.Sigmoid(self.conv5(out))

        out=nn.functional.interpolate(out,scale_factor=2)   #a*b
        return out


class Net(nn.Module):
    def __init__(self,num_class):
        super(Net,self).__init__()

        self.lowLevel=LowLevelNet()
        self.midLevel=MidLevelNet()
        self.globalLevel=GlobalNet()

        #融合后
        self.fuse=nn.Conv2d(512,256,1,1,0)#

        self.classNet=ClassNet(num_class)
        self.colorize=ColorizeNet()

    def forward(self,x):
        lowNetOut=self.lowLevel(x)

        midNetOut=self.midLevel(lowNetOut)
        fusionIn,classIn=self.globalLevel(lowNetOut)

        classOut=self.classNet(classIn)

        #下面是融合层 之后进入着色层，输出out



