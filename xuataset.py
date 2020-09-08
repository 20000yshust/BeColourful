import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy
from torchvision import transforms
import colorsys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


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
                tmp = os.path.join(root2, f)
                tmp = Image.open(tmp)
                imgs1.append(numpy.asarray(tmp))
                label.append(i)
            i = i+1
        self.imgs = imgs1
        self.label = label
        self.transforms = transform

    # def __getitem__(self, index):
    #     img_1 = self.imgs[index]
    #     label = self.label[index]
    #
    #     img = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
    #     img = cv2.resize(img, (img_size, img_size))
    #     labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    #
    #     plt.imshow(labimg[0])
    #     plt.show()
    #
    #     pil_img = numpy.array(labimg)
    #     if self.transforms:
    #         data = self.transforms(pil_img)
    #     else:
    #         pil_img = numpy.asarray(pil_img)
    #         data = torch.from_numpy(pil_img)
    #     return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_data = MySet(r'data\train')

    BATCH = 16
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=False, num_workers=0)
