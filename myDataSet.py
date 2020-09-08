import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from PIL import Image
import numpy
from torchvision import transforms
import colorsys
import cv2


img_size=224

transform = transforms.Compose([
    transforms.ToTensor() # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

#定义自己的数据集合
class MySet(Data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        imgs1 = []
        label = []
        i=0
        folders = os.listdir(root)
        for k in folders:
            root2 = os.path.join(root, k)
            jpg = os.listdir(root2)
            for f in jpg:
                e = os.path.join(root2, f)
                imgs1.append(e)
                label.append(i)
            i=i+1
        self.imgs = imgs1
        self.label = label
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.label[index]
        label = torch.from_numpy(numpy.array(label))

        img_1 = Image.open(img_path)
        img = cv2.cvtColor(numpy.asarray(img_1), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (img_size, img_size))
        labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        pil_img = numpy.array(labimg)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = torch.from_numpy(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)

