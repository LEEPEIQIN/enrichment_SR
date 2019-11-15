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
    



class RDB(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(RDB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._initialize_weights()
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)
        init.orthogonal_(self.conv2.weight)
        init.orthogonal_(self.conv3.weight)
        init.orthogonal_(self.conv4.weight)
        init.orthogonal_(self.conv5.weight)
    
         
class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.pooling=nn.AdaptiveAvgPool2d(1)
        self.RB_1 = nn.Conv1d(64,16,1,1)
        self.RB_2 = nn.Conv1d(16,64,1,1)
        self._initialize_weights()
    def forward(self, x):
        x=self.pooling(x)
        x=F.relu(self.RB_1(x.squeeze(2)))
        x=torch.sigmoid(self.RB_2(x))
        return x.unsqueeze(3)
    def _initialize_weights(self):
        init.orthogonal_(self.RB_1.weight)
        init.orthogonal_(self.RB_2.weight)


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.rdb=RDB()
        self.ca=CA()
    def forward(self,x):
        keep=x
        x=self.rdb(x)
        c=self.ca(x)
        x=x*c*0.2
        x+=keep
        return x

class RRDB(nn.Module):
    def __init__(self):
        super(RRDB, self).__init__()
        self.block_1=Block()
        self.block_2=Block()
        self.block_3=Block()
    def forward(self,x):
        keep=x
        x=self.block_1(x)
        x=self.block_2(x)
        x=self.block_3(x)
        x=keep+x*0.2
        return x
    
class Net_block(nn.Module):
    def __init__(self):
        super(Net_block, self).__init__()
        self.int = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.rrdb_1=RRDB()
        self.rrdb_2=RRDB()
        self.rrdb_3=RRDB()
        self.rrdb_4=RRDB()
        self.rrdb_5=RRDB()
        self.rrdb_6=RRDB()
        self.end1=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.end2=nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.end3=nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.to_image=nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._initialize_weights()
        
    def forward(self,x):
        x=self.int(x)
        keep=x
        x=self.rrdb_1(x)
        x=self.rrdb_2(x)
        x=self.rrdb_3(x)
        x=self.rrdb_4(x)
        x=self.rrdb_5(x)
        x=self.rrdb_6(x)
        x=self.end1(x)
        x+=keep
        x=self.lrelu(self.end2(x))
        x=self.lrelu(self.end3(x))
        x=self.to_image(x)
        return x
    def _initialize_weights(self):
        init.orthogonal_(self.int.weight)
        init.orthogonal_(self.end1.weight)
        init.orthogonal_(self.end2.weight)
        init.orthogonal_(self.end3.weight)
        init.orthogonal_(self.to_image.weight)
        
        
        
        
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
            HR,LR=generator("2_HR",LR_URL,16,128)
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
                del HR_test,LR_test,temp
        PSNR=PSNR/n_test
        print(PSNR.item())
        del PSNR,n_test,temp_HR_2,temp_LR_2
        
    p=[]
    optimizer=torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(b):
        running_loss=0.0
        for i in range(1000):
            HR,LR=generator("2_HR",LR_URL,16,128)
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
                del HR_test,LR_test,temp
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
            HR,LR=generator("2_HR",LR_URL,16,128)
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
                del HR_test,LR_test,temp
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



         
