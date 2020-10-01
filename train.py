from model import *
from myDataSet import *
import numpy as np
from torch.utils.data import DataLoader
from utils import *
import torchvision.transforms as transforms
from tqdm import tqdm

GPU = False
BATCH = 16
lamda_cls = 1
lamda_color = 1
LR = 1e-6
EPOCH = 16
save_version = 0
reuse_version = 0
reuse = False
train_file = "./data/train"
test_file = "./data/test"

trainSet = MySet(train_file)
testSet = MySet(test_file)

#LOSS

CLASSIFY_loss = nn.CrossEntropyLoss()
COLOR_loss = nn.MSELoss()

model = Net(num_classes=6)

if reuse:
    try:
        model = resume_checkpoint(model,checkpoint_dirname='model', version=reuse_version)
    except:
        pass

opt = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-6)

if GPU:
    model = model.cuda(0)

trainLoader=DataLoader(trainSet, batch_size=BATCH, shuffle=True,num_workers=0)
testLoader=DataLoader(testSet, batch_size=BATCH, shuffle=True,num_workers=0)

for epoch in range(EPOCH):
    model.train()
    train_acc = 0.
    epoch_cls_loss = 0.
    epoch_color_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(tqdm(trainLoader, ncols=50)):


        if GPU is True:
            batch_imgs = imgs.cuda(0).float()
            batch_labels = labels.cuda().long()
        else:
            batch_imgs = imgs.float()
            batch_labels = labels.long()

        inBatch_imgs = batch_imgs[:, 0, :, :].unsqueeze(1)
        pre_imgs, pre_labels = model(inBatch_imgs)

        pre_imgs = torch.reshape(pre_imgs, (len(batch_imgs), -1))
        batch_imgs = torch.reshape(batch_imgs[:, 1:, :, :], (len(batch_imgs), -1))
        cls_loss = CLASSIFY_loss(pre_labels, batch_labels)
        color_loss = COLOR_loss(pre_imgs, batch_imgs)

        LOSS = lamda_cls * cls_loss + lamda_color * color_loss

        opt.zero_grad()
        LOSS.backward()
        opt.step()
        epoch_cls_loss += cls_loss.item()
        epoch_color_loss += color_loss.item()

        _, pred = pre_labels.max(1)
        #print(pred)###########################
        num_correct = (pred.long() == batch_labels.long()).sum().item()
        acc = num_correct / len(batch_labels)
        train_eval = acc
        train_acc += acc

    train_acc = train_acc / len(trainLoader)
    epoch_cls_loss /= len(trainLoader)
    epoch_color_loss /= len(trainLoader)
    print('[Epoch:{}/{},Acc:{:.1f}%,Loss:{} colorLoss:{} clsLoss:{}]'.format(
        epoch, EPOCH, train_acc * 100, lamda_color * epoch_color_loss + lamda_cls * epoch_cls_loss, epoch_mask_loss,
        epoch_cls_loss
    ))
    print('lamda: color is {} and cls is {}'.format(lamda_color,lamda_cls))
    model.eval()
    eval_acc = 0.
    for batch_idx, (imgs, labels) in enumerate(tqdm(testLoader, ncols=50)):
        if GPU is True:
            batch_imgs = imgs.cuda(0).float()
            batch_labels = labels.cuda().long()
        else:
            batch_imgs = imgs.float()
            batch_labels = labels.long()

        pre_imgs, pre_labels = model(batch_imgs)

        pre_imgs = torch.reshape(pre_imgs, (len(batch_imgs), -1))
        batch_imgs = torch.reshape(batch_imgs, (len(batch_imgs), -1))
        cls_loss = CLASSIFY_loss(pre_labels, batch_labels)
        color_loss = COLOR_loss(pre_imgs, batch_imgs)
        _,pred = pre_labels.max(1)
        num_correct = (pred.long() == batch_labels.long()).sum().item()
        acc = num_correct / len(batch_labels)
        eval_acc += acc
    epoch_cls_loss /= len(testLoader)
    epoch_color_loss /= len(testLoader)
    eval_acc = eval_acc / len(testLoader)
    print('Acc:{:.1f}%'.format(eval_acc * 100))
save_checkpoint(model, checkpoint_dirname='model', version=save_version)