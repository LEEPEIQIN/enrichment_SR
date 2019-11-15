#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import common
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.L1Loss()
cwd = os.getcwd()
os.chdir('module')
mwd=os.getcwd()
os.chdir(cwd)
#######################################################################################################
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

net1=common.Net_block().to(device)

optimizer=torch.optim.Adam(net1.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(10):
    running_loss=0.0
    for i in range(1000):
        HR,LR=common.generator("2_HR","2_LR",16,128)
        HR=HR.to(device)
        LR=LR.to(device)
        optimizer.zero_grad()
        outputs=net1(LR)
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
             LR_1=net1(LR_tl).data
             LR_2=net1(LR_tr).data
             LR_3=net1(LR_bl).data
             LR_4=net1(LR_br).data
             del LR_tl,LR_tr,LR_bl,LR_br
             temp1=torch.cat((LR_1,LR_2),3)
             del LR_1,LR_2
             temp2=torch.cat((LR_3,LR_4),3)
             del LR_3,LR_4
             temp=torch.cat((temp1,temp2),2).squeeze()
             del temp1,temp2
             temp[temp>1]=1
             temp[temp<0]=0
             PSNR+=common.psnr(temp,HR_test)
             del HR_test,LR_test,temp
    PSNR=PSNR/n_test
    print(PSNR.item())
    del PSNR,n_test,temp_HR_2,temp_LR_2

p1=common.net_train(net1,"2_LR","2_LR_test","2_LR_afternet1","2_LR_test_afternet1","net1",10,20,20)
del net1
############################################################################################

net2=common.Net_block()
net2.load_state_dict(torch.load('module/net1.pt'))
net2.train()
net2.to(device)

p2=common.net_train(net2,"temp_image/2_LR_afternet1","temp_image/2_LR_test_afternet1","2_LR_afternet2","2_LR_test_afternet2","net2",10,20,20)
del net2
#########################################################################################

net3=common.Net_block()
net3.load_state_dict(torch.load('module/net2.pt'))
net3.train()
net3.to(device)         

p3=common.net_train(net3,"temp_image/2_LR_afternet2","temp_image/2_LR_test_afternet2","2_LR_afternet3","2_LR_test_afternet3","net3",10,20,20)
del net3
###############################################################################################

net_combine1=common.Net_combine()

net_combine1.net1.load_state_dict(torch.load('module/net1.pt'))
net_combine1.net2.load_state_dict(torch.load('module/net2.pt'))
net_combine1.net3.load_state_dict(torch.load('module/net3.pt'))
net_combine1.train()
net_combine1.to(device)

pc1=common.net_train(net_combine1,"2_LR","2_LR_test","2_LR_afterc1","2_LR_test_afterc1","net_c1",10,20,20)
del net_combine1
############################################################################################################
net_combine2=common.Net_combine()

net_combine2.load_state_dict(torch.load('module/net_c1.pt'))
net_combine2.train()
net_combine2.to(device)
pc2=common.net_train(net_combine2,"temp_imgae/2_LR_afterc1","temp_image/2_LR_test_afterc1","2_LR_afterc2","2_LR_test_afterc2","net_c2",10,20,20)
del net_combine2
########################################################################################################
net_combine3=common.Net_combine()

net_combine3.load_state_dict(torch.load('module/net_c2.pt'))
net_combine3.train()
net_combine3.to(device)
pc2=common.net_train(net_combine3,"temp_imgae/2_LR_afterc2","temp_image/2_LR_test_afterc2","2_LR_afterc3","2_LR_test_afterc3","net_c3",10,20,20)
del net_combine3
###########################################################################################################
net_final1=common.Net_final()

net_final1.net_1.load_state_dict(torch.load('module/net_c1.pt'))
net_final1.net_2.load_state_dict(torch.load('module/net_c2.pt'))
net_final1.net_3.load_state_dict(torch.load('module/net_c3.pt'))
net_final1.train()
net_final1.to(device)

pc1=common.net_train(net_final1,"2_LR","2_LR_test","2_LR_afterf1","2_LR_test_afterf1","net_f1",10,20,20)
del net_final1
