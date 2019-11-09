#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
from random import randrange
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import torchvision.transforms as transforms

def generator(HR_URL,LR_URL,batch_size,image_size):
    #given:
    channel=3
    #genereting:
    HR_set=torchvision.datasets.ImageFolder(root=HR_URL)
    LR_set=torchvision.datasets.ImageFolder(root=LR_URL)
    HR_torch=torch.zeros([batch_size,channel,image_size,image_size])
    LR_torch=torch.zeros([batch_size,channel,image_size,image_size])
    for i in range(batch_size):
        img_number=randrange(len(HR_set))
        HR_img=Image.open(HR_set.imgs[img_number][0])
        LR_img=Image.open(LR_set.imgs[img_number][0])
        y,x=HR_img.size
        X=randrange(0,x-image_size)
        Y=randrange(0,y-image_size)
        HR_img=torchvision.transforms.functional.crop(HR_img, X, Y, image_size, image_size)
        LR_img=torchvision.transforms.functional.crop(LR_img, X, Y, image_size, image_size)
        #whether rotate and whether flipHR_:
        p_f=randrange(2)
        d_r=randrange(4)*90
        HR_img=torchvision.transforms.functional.rotate(HR_img,d_r)
        LR_img=torchvision.transforms.functional.rotate(LR_img,d_r)
        if (p_f==1):
            HR_img=HR_img.transpose(Image.FLIP_LEFT_RIGHT)
            LR_img=LR_img.transpose(Image.FLIP_LEFT_RIGHT)
        ##tensor defined:
        HR_torch[i,:,:,:]=torchvision.transforms.functional.to_tensor(HR_img)
        LR_torch[i,:,:,:]=torchvision.transforms.functional.to_tensor(LR_img)
    ans=[HR_torch,LR_torch]
    return ans


def psnr(target, ref):
    # target:目标图像  ref:参考图像 
    # assume RGB image
    diff = ref- target
    diff = diff.reshape(-1)
    rmse = torch.sqrt((diff ** 2.).mean())
    return 20*torch.log10(1.0/rmse)
    



class Net_block(nn.Module):
    def __init__(self):
        super(Net_block, self).__init__()
        self.pooling=nn.AdaptiveAvgPool2d(1)
        #initial cnn:
        self.int = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        #RB_1:
        self.RB_1_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_1_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_1_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_1_1d2 = nn.Conv1d(16,64,1,1)
        #RB_2:
        self.RB_2_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_2_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_2_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_2_1d2 = nn.Conv1d(16,64,1,1)
        #RB_3:
        self.RB_3_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_3_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_3_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_3_1d2 = nn.Conv1d(16,64,1,1)
        #RB_4:
        self.RB_4_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_4_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_4_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_4_1d2 = nn.Conv1d(16,64,1,1)
        #RB_5:
        self.RB_5_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_5_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_5_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_5_1d2 = nn.Conv1d(16,64,1,1)
        #
        self.c1=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #RB_6:
        self.RB_6_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_6_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_6_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_6_1d2 = nn.Conv1d(16,64,1,1)
        #RB_7:
        self.RB_7_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_7_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_7_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_7_1d2 = nn.Conv1d(16,64,1,1)
        #RB_8:
        self.RB_8_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_8_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_8_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_8_1d2 = nn.Conv1d(16,64,1,1)
        #RB_9:
        self.RB_9_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_9_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_9_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_9_1d2 = nn.Conv1d(16,64,1,1)
        #RB_10:
        self.RB_10_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_10_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_10_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_10_1d2 = nn.Conv1d(16,64,1,1)
        #
        self.c2=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #RB_11:
        self.RB_11_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_11_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_11_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_11_1d2 = nn.Conv1d(16,64,1,1)
        #RB_12:
        self.RB_12_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_12_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_12_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_12_1d2 = nn.Conv1d(16,64,1,1)
        #RB_13:
        self.RB_13_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_13_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_13_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_13_1d2 = nn.Conv1d(16,64,1,1)
        #RB_14:
        self.RB_14_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_14_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_14_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_14_1d2 = nn.Conv1d(16,64,1,1)
        #RB_15:
        self.RB_15_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_15_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_15_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_15_1d2 = nn.Conv1d(16,64,1,1)
        #
        self.c3=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #RB_16:
        self.RB_16_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_16_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_16_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_16_1d2 = nn.Conv1d(16,64,1,1)
        #RB_17:
        self.RB_17_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_17_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_17_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_17_1d2 = nn.Conv1d(16,64,1,1)
        #RB_18:
        self.RB_18_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_18_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_18_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_18_1d2 = nn.Conv1d(16,64,1,1)
        #RB_19:
        self.RB_19_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_19_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_19_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_19_1d2 = nn.Conv1d(16,64,1,1)
        #RB_20:
        self.RB_20_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_20_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_20_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_20_1d2 = nn.Conv1d(16,64,1,1)
        #
        self.c4=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #end cnn:
        self.end_RB=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.toR=nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self._initialize_weights()
    def forward(self, x):
        x=self.int(x)
        R=x
        ############################################################
        r1=x
        #RB_1
        R_1=x
        x=F.relu(self.RB_1_conv1(x))
        x=self.RB_1_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_1_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_1_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_1
        #RB_2
        R_2=x
        x=F.relu(self.RB_2_conv1(x))
        x=self.RB_2_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_2_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_2_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_2
        #RB_3
        R_3=x
        x=F.relu(self.RB_3_conv1(x))
        x=self.RB_3_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_3_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_3_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_3
        #RB_4
        R_4=x
        x=F.relu(self.RB_4_conv1(x))
        x=self.RB_4_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_4_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_4_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_4
        #RB_5
        R_5=x
        x=F.relu(self.RB_5_conv1(x))
        x=self.RB_5_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_5_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_5_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_5
        #
        x=self.c1(x)
        x+=r1
        ##############################################################
        r2=x
        #RB_6
        R_6=x
        x=F.relu(self.RB_6_conv1(x))
        x=self.RB_6_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_6_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_6_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_6
        #RB_7
        R_7=x
        x=F.relu(self.RB_7_conv1(x))
        x=self.RB_7_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_7_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_7_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_7
        #RB_8
        R_8=x
        x=F.relu(self.RB_8_conv1(x))
        x=self.RB_8_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_8_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_8_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_8
        #RB_9
        R_9=x
        x=F.relu(self.RB_9_conv1(x))
        x=self.RB_9_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_9_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_9_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_9
        #RB_10
        R_10=x
        x=F.relu(self.RB_10_conv1(x))
        x=self.RB_10_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_10_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_10_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_10
        #
        x=self.c2(x)
        x+=r2
        #######################################################################
        r3=x
        #RB_11
        R_11=x
        x=F.relu(self.RB_11_conv1(x))
        x=self.RB_11_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_11_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_11_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_11
        #RB_12
        R_12=x
        x=F.relu(self.RB_12_conv1(x))
        x=self.RB_12_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_12_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_12_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_12
        #RB_13
        R_13=x
        x=F.relu(self.RB_13_conv1(x))
        x=self.RB_13_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_13_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_13_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_13
        #RB_14
        R_14=x
        x=F.relu(self.RB_14_conv1(x))
        x=self.RB_14_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_14_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_14_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_14
        #RB_15
        R_15=x
        x=F.relu(self.RB_15_conv1(x))
        x=self.RB_15_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_15_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_15_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_15
        #
        x=self.c3(x)
        x+=r3
        ##################################################################
        r4=x
        #RB_16
        R_16=x
        x=F.relu(self.RB_16_conv1(x))
        x=self.RB_16_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_16_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_16_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_16
        #RB_17
        R_17=x
        x=F.relu(self.RB_17_conv1(x))
        x=self.RB_17_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_17_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_17_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_17
        #RB_18
        R_18=x
        x=F.relu(self.RB_18_conv1(x))
        x=self.RB_18_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_18_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_18_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_18
        #RB_19
        R_19=x
        x=F.relu(self.RB_19_conv1(x))
        x=self.RB_19_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_19_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_19_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_19
        #RB_20
        R_20=x
        x=F.relu(self.RB_20_conv1(x))
        x=self.RB_20_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_20_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_20_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_20
        #
        x=self.c4(x)
        x+=r4
        #################################################################
        
        #end:
        x=self.end_RB(x)
        x+=R
        x=self.toR(x)
        return x
    def _initialize_weights(self):
        init.orthogonal_(self.int.weight)
        init.orthogonal_(self.c1.weight)
        init.orthogonal_(self.c2.weight)
        init.orthogonal_(self.c3.weight)
        init.orthogonal_(self.c4.weight)
        #RB_1
        init.orthogonal_(self.RB_1_conv1.weight)
        init.orthogonal_(self.RB_1_conv2.weight)
        init.orthogonal_(self.RB_1_1d1.weight)
        init.orthogonal_(self.RB_1_1d2.weight)
        #RB_2
        init.orthogonal_(self.RB_2_conv1.weight)
        init.orthogonal_(self.RB_2_conv2.weight)
        init.orthogonal_(self.RB_2_1d1.weight)
        init.orthogonal_(self.RB_2_1d2.weight)
        #RB_3
        init.orthogonal_(self.RB_3_conv1.weight)
        init.orthogonal_(self.RB_3_conv2.weight)
        init.orthogonal_(self.RB_3_1d1.weight)
        init.orthogonal_(self.RB_3_1d2.weight)
        #RB_4
        init.orthogonal_(self.RB_4_conv1.weight)
        init.orthogonal_(self.RB_4_conv2.weight)
        init.orthogonal_(self.RB_4_1d1.weight)
        init.orthogonal_(self.RB_4_1d2.weight)
        #RB_5
        init.orthogonal_(self.RB_5_conv1.weight)
        init.orthogonal_(self.RB_5_conv2.weight)
        init.orthogonal_(self.RB_5_1d1.weight)
        init.orthogonal_(self.RB_5_1d2.weight)
        #RB_6
        init.orthogonal_(self.RB_6_conv1.weight)
        init.orthogonal_(self.RB_6_conv2.weight)
        init.orthogonal_(self.RB_6_1d1.weight)
        init.orthogonal_(self.RB_6_1d2.weight)
        #RB_7
        init.orthogonal_(self.RB_7_conv1.weight)
        init.orthogonal_(self.RB_7_conv2.weight)
        init.orthogonal_(self.RB_7_1d1.weight)
        init.orthogonal_(self.RB_7_1d2.weight)
        #RB_8
        init.orthogonal_(self.RB_8_conv1.weight)
        init.orthogonal_(self.RB_8_conv2.weight)
        init.orthogonal_(self.RB_8_1d1.weight)
        init.orthogonal_(self.RB_8_1d2.weight)
        #RB_9
        init.orthogonal_(self.RB_9_conv1.weight)
        init.orthogonal_(self.RB_9_conv2.weight)
        init.orthogonal_(self.RB_9_1d1.weight)
        init.orthogonal_(self.RB_9_1d2.weight)
        #RB_10
        init.orthogonal_(self.RB_10_conv1.weight)
        init.orthogonal_(self.RB_10_conv2.weight)
        init.orthogonal_(self.RB_10_1d1.weight)
        init.orthogonal_(self.RB_10_1d2.weight)
        #RB_11
        init.orthogonal_(self.RB_11_conv1.weight)
        init.orthogonal_(self.RB_11_conv2.weight)
        init.orthogonal_(self.RB_11_1d1.weight)
        init.orthogonal_(self.RB_11_1d2.weight)
        #RB_12
        init.orthogonal_(self.RB_12_conv1.weight)
        init.orthogonal_(self.RB_12_conv2.weight)
        init.orthogonal_(self.RB_12_1d1.weight)
        init.orthogonal_(self.RB_12_1d2.weight)
        #RB_13
        init.orthogonal_(self.RB_13_conv1.weight)
        init.orthogonal_(self.RB_13_conv2.weight)
        init.orthogonal_(self.RB_13_1d1.weight)
        init.orthogonal_(self.RB_13_1d2.weight)
        #RB_14
        init.orthogonal_(self.RB_14_conv1.weight)
        init.orthogonal_(self.RB_14_conv2.weight)
        init.orthogonal_(self.RB_14_1d1.weight)
        init.orthogonal_(self.RB_14_1d2.weight)
        #RB_15
        init.orthogonal_(self.RB_15_conv1.weight)
        init.orthogonal_(self.RB_15_conv2.weight)
        init.orthogonal_(self.RB_15_1d1.weight)
        init.orthogonal_(self.RB_15_1d2.weight)
        #RB_16
        init.orthogonal_(self.RB_16_conv1.weight)
        init.orthogonal_(self.RB_16_conv2.weight)
        init.orthogonal_(self.RB_16_1d1.weight)
        init.orthogonal_(self.RB_16_1d2.weight)
        #RB_17
        init.orthogonal_(self.RB_17_conv1.weight)
        init.orthogonal_(self.RB_17_conv2.weight)
        init.orthogonal_(self.RB_17_1d1.weight)
        init.orthogonal_(self.RB_17_1d2.weight)
        #RB_18
        init.orthogonal_(self.RB_18_conv1.weight)
        init.orthogonal_(self.RB_18_conv2.weight)
        init.orthogonal_(self.RB_18_1d1.weight)
        init.orthogonal_(self.RB_18_1d2.weight)
        #RB_19
        init.orthogonal_(self.RB_19_conv1.weight)
        init.orthogonal_(self.RB_19_conv2.weight)
        init.orthogonal_(self.RB_19_1d1.weight)
        init.orthogonal_(self.RB_19_1d2.weight)
        #RB_20
        init.orthogonal_(self.RB_20_conv1.weight)
        init.orthogonal_(self.RB_20_conv2.weight)
        init.orthogonal_(self.RB_20_1d1.weight)
        init.orthogonal_(self.RB_20_1d2.weight)
     
        init.orthogonal_(self.end_RB.weight) 
        init.orthogonal_(self.toR.weight)
    
         

class Net_combine(nn.Module):
    def __init__(self):
        super(Net_combine, self).__init__()
        self.net1=Net_block()
        self.net2=Net_block()
        self.net3=Net_block()
    def forward(self, x):
        x=self.net1(x)
        x[x>1]=1
        x[x<0]=0
        x=self.net2(x)
        x[x>1]=1
        x[x<0]=0
        x=self.net3(x)
        return x

class Net_final(nn.Module):
    def __init__(self):
        super(Net_final, self).__init__()
        self.net_1=Net_combine()
        self.net_2=Net_combine()
        self.net_3=Net_combine()
    def forward(self,x):
        x=self.net_1(x)
        x[x>1]=1
        x[x<0]=0
        x=self.ne_t2(x)
        x[x>1]=1
        x[x<0]=0
        x=self.net_3(x)
        return x


    
    
def generate_image(net,first,second): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cwd=os.getcwd()
    LR_URL=first
    LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())    
    LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
    LR_loader=iter(LR_loader)
    os.chdir("temp_image")
    os.mkdir(second)
    os.chdir(second)
    os.mkdir('2')
    os.chdir('2')
    wd=os.getcwd()
    os.chdir(cwd)
    with torch.no_grad():
        for i in range(len(LR_set)):
            LR=LR_loader.next()[0]
            _,_,x,y=LR.size()
            temp=torch.nn.functional.unfold(LR,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
            LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
            LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
            LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
            LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
            LR_1=net(LR_tl).data
            LR_2=net(LR_tr).data
            LR_3=net(LR_bl).data
            LR_4=net(LR_br).data
            temp1=torch.cat((LR_1,LR_2),3)
            temp2=torch.cat((LR_3,LR_4),3)
            temp=torch.cat((temp1,temp2),2).squeeze()
            temp[temp>1]=1
            temp[temp<0]=0
            LR_after=temp.cpu()
            LR_after = transforms.ToPILImage()(LR_after)
            os.chdir(wd)
            if i<9:
                LR_after.save(second+"_00"+str(i+1)+".png","PNG")
            elif i<99:
                LR_after.save(second+"_0"+str(i+1)+".png","PNG")
            else:
                LR_after.save(second+"_"+str(i+1)+".png","PNG")
            os.chdir(cwd)
    return True
            
    

def net_train(net, LR_URL, LR_test_URL, LR_name, LR_test_name, net_name, a,b,c):
    test_LR = torchvision.datasets.ImageFolder(root=LR_test_URL, transform=transforms.ToTensor())
    LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
    test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
    HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    cwd = os.getcwd()
    os.chdir('module')
    mwd=os.getcwd()
    os.chdir(cwd)

    optimizer=torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(a):
        running_loss=0.0
        for i in range(1000):
            HR,LR=generator("2_HR",LR_URL,16,120)
            HR=HR.to(device)
            LR=LR.to(device)
            optimizer.zero_grad()
            outputs=net(LR)
            Loss=criterion(outputs,HR)
            Loss.backward()
            optimizer.step()
            running_loss+=Loss.item()
            
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            del HR,LR,Loss,outputs
        n_test=15
        PSNR=0.0
        temp_HR_2=iter(HR_2_test)
        temp_LR_2=iter(LR_2_test)
        with torch.no_grad():
            for t in range(n_test):
                HR_test=temp_HR_2.next()[0].squeeze().to(device)
                LR_test=temp_LR_2.next()[0]
                _,_,x,y=LR_test.size()
                temp=torch.nn.functional.unfold(LR_test,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
                LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
                del temp
                LR_1=net(LR_tl).data
                LR_2=net(LR_tr).data
                LR_3=net(LR_bl).data
                LR_4=net(LR_br).data
                del LR_tl,LR_tr,LR_bl,LR_br
                temp1=torch.cat((LR_1,LR_2),3)
                del LR_1,LR_2
                temp2=torch.cat((LR_3,LR_4),3)
                del LR_3,LR_4
                temp=torch.cat((temp1,temp2),2).squeeze()
                del temp1,temp2
                temp[temp>1]=1
                temp[temp<0]=0
                PSNR+=psnr(temp,HR_test)
                del HR_test,LR_test,outputs
        PSNR=PSNR/n_test
        print(PSNR.item())
        del PSNR,n_test,temp_HR_2,temp_LR_2
        
    p=[]
    optimizer=torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(b):
        running_loss=0.0
        for i in range(1000):
            HR,LR=generator("2_HR",LR_URL,16,120)
            HR=HR.to(device)
            LR=LR.to(device)
            optimizer.zero_grad()
            outputs=net(LR)
            outputs[outputs>1]=1
            outputs[outputs<0]=0
            Loss=criterion(outputs,HR)
            Loss.backward()
            optimizer.step()
            running_loss+=Loss.item()
            
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            del HR,LR,Loss,outputs
        n_test=15
        PSNR=0.0
        temp_HR_2=iter(HR_2_test)
        temp_LR_2=iter(LR_2_test)
        with torch.no_grad():
            for t in range(n_test):
                HR_test=temp_HR_2.next()[0].squeeze().to(device)
                LR_test=temp_LR_2.next()[0]
                _,_,x,y=LR_test.size()
                temp=torch.nn.functional.unfold(LR_test,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
                LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
                del temp
                LR_1=net(LR_tl).data
                LR_2=net(LR_tr).data
                LR_3=net(LR_bl).data
                LR_4=net(LR_br).data
                del LR_tl,LR_tr,LR_bl,LR_br
                temp1=torch.cat((LR_1,LR_2),3)
                del LR_1,LR_2
                temp2=torch.cat((LR_3,LR_4),3)
                del LR_3,LR_4
                temp=torch.cat((temp1,temp2),2).squeeze()
                del temp1,temp2
                temp[temp>1]=1
                temp[temp<0]=0
                PSNR+=psnr(temp,HR_test)
                del HR_test,LR_test,outputs
        PSNR=PSNR/n_test
        print(PSNR.item())
        p.append(PSNR.item())
        del PSNR,n_test,temp_HR_2,temp_LR_2
        os.chdir(mwd)
        torch.save(net.state_dict(), net_name+".pt")
        os.chdir(cwd)
        
    optimizer=torch.optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(c):
        running_loss=0.0
        for i in range(1000):
            HR,LR=generator("2_HR",LR_URL,16,120)
            HR=HR.to(device)
            LR=LR.to(device)
            optimizer.zero_grad()
            outputs=net(LR)
            outputs[outputs>1]=1
            outputs[outputs<0]=0
            Loss=criterion(outputs,HR)
            Loss.backward()
            optimizer.step()
            running_loss+=Loss.item()
            
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            del HR,LR,Loss,outputs
        n_test=15
        PSNR=0.0
        temp_HR_2=iter(HR_2_test)
        temp_LR_2=iter(LR_2_test)
        with torch.no_grad():
            for t in range(n_test):
                HR_test=temp_HR_2.next()[0].squeeze().to(device)
                LR_test=temp_LR_2.next()[0]
                _,_,x,y=LR_test.size()
                temp=torch.nn.functional.unfold(LR_test,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
                LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
                LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
                del temp
                LR_1=net(LR_tl).data
                LR_2=net(LR_tr).data
                LR_3=net(LR_bl).data
                LR_4=net(LR_br).data
                del LR_tl,LR_tr,LR_bl,LR_br
                temp1=torch.cat((LR_1,LR_2),3)
                del LR_1,LR_2
                temp2=torch.cat((LR_3,LR_4),3)
                del LR_3,LR_4
                temp=torch.cat((temp1,temp2),2).squeeze()
                del temp1,temp2
                temp[temp>1]=1
                temp[temp<0]=0
                PSNR+=psnr(temp,HR_test)
                del HR_test,LR_test,outputs
        PSNR=PSNR/n_test
        print(PSNR.item())
        p.append(PSNR.item())
        del PSNR,n_test,temp_HR_2,temp_LR_2
        os.chdir(mwd)
        torch.save(net.state_dict(), net_name+".pt")
        os.chdir(cwd)
    print (net_name + "finished")
    print(p)

    generate_image(net,LR_URL,LR_name)
    generate_image(net,LR_test_URL,LR_test_name)
    return p



         
