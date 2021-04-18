# DPIR training code by Nick.Young
import os
import torch
import torch.nn as nn
import dataset
from models.network_unet import UNetRes as DRUNet
from models.network_unet import DnCNN
from torch.utils.data import DataLoader
import random
import math
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Set to 1 on the server | Set to 0 on the laptop


# Training Settings
##
## Mode switch
train_model = True  
create_model = True  
select_model = 'DPIR'  # 'DnCNN' | 'DPIR'
Load_Log_name = '/Test-Log1.pth'
Save_Log_name = '/Test-Log1.pth'

create_patch = False  
create_noise =  False  
create_PairedNoise = False  
##
## Hyperparameter
learning_rate = (1e-4)  
total_epoch_num = 300  
LrStep = 100  # Step of decreasing learning rate
LrRaio = 0.1  # Ratio to decrease
DrawStep = 2  # Step of appending x[],y[], and drawing the temporary loss figure
Batch_Size = 4  
##
## File Path&Name
rpath = '/share21/home/renwei/projects/DPIR/'
TrainSet_path = rpath + '/train_sets/set12/'  # training set path
TrainNoise_path = rpath + '/train_sets/noise-512/'
Log_path = rpath + 'model_zoo/'  # Model file path
LossFigure_path = rpath + '/results/LossFigure/'
TestSet_path = rpath + '/results/SPnoise'  # testing set path


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(
            -0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def drawFig(loss_x, loss_y, Epoch):
    lossfig = plt.plot(loss_x, loss_y)  #画出最终的loss曲线
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if type(Epoch) == str:
        plt.savefig(LossFigure_path + '/Epoch-Tmp.png')  #保存loss曲线
    if type(Epoch) == int:
        plt.title('LossFigure of Epoch:%d' % (Epoch + 1))
        plt.savefig(LossFigure_path + '/Epoch-Final.png')  #保存loss曲线

    plt.close()


def get_noiseONE_Map(SPlevel):  # generate ONE 128*128 noise map
    SPpoints = random.sample(range(128 * 128), int(
        128 * 128 * SPlevel))  # generate SP noise points
    noiseONE_Map = torch.zeros([128, 128])
    for sppoints in SPpoints:
        noiseONE_Mapx = sppoints % 128  # generate x
        noiseONE_Mapy = sppoints // 128  # generate y
        noiseONE_Map[noiseONE_Mapx, noiseONE_Mapy] = random.choice([1, -1])
    return noiseONE_Map


def main_DPIR():

    model = DRUNet(in_nc=1,
                   out_nc=1,
                   nc=[64, 128, 256, 512],
                   nb=4,
                   act_mode='R',
                   downsample_mode='strideconv',
                   upsample_mode='convtranspose')

    model = model.cuda()

    # criterion = nn.SmoothL1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    if create_model:
        model.apply(weights_init_kaiming)
    else:
        model.load_state_dict(torch.load(Log_path + Load_Log_name))

    dataset_train = dataset.Dataset(
        train=True)  #dataset.Dataset 读取了 'train.h5' 文件
    Batch_dataset = DataLoader(dataset=dataset_train,
                               num_workers=4,
                               batch_size=Batch_Size,
                               shuffle=True)

    loss_x = [0]
    loss_y = [0]
    Map_nums = range(150)  #* use 5%~95%
    lr = learning_rate

    for epoch in range(total_epoch_num):

        print('\n\n\n')

        if epoch % LrStep == 0 and epoch > 0:
            lr = lr * (LrRaio)
            print('Halved learning rate')

        print('[Epoch:%d] Learning rate=' % (epoch + 1), lr, '\n')

        Map_num = random.choice(Map_nums)
        noiseEPOCH_Map = np.load(TrainNoise_path + '/noiseMap_%d.npy' %
                                 (Map_num))
        print('[Epoch:%d] noiseMap_%d has been selected\n' %
              (epoch + 1, Map_num))
        noiseEPOCH_Map1 = noiseEPOCH_Map[0, :, 0:128, 0:128]
        noiseEPOCH_Map2 = noiseEPOCH_Map[0, :, 128:256, 0:128]
        noiseEPOCH_Map3 = noiseEPOCH_Map[0, :, 0:128, 128:256]
        noiseEPOCH_Map4 = noiseEPOCH_Map[0, :, 128:256, 128:256]
        noiseEPOCH_Map = np.stack((noiseEPOCH_Map1, noiseEPOCH_Map2,
                                   noiseEPOCH_Map3, noiseEPOCH_Map4),
                                  axis=0)
        noiseEPOCH_Map = torch.Tensor(noiseEPOCH_Map)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        for i, Batch_sample in enumerate(Batch_dataset):  #【+】遍历每个批次
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # Method 1:加噪
            img_x = Batch_sample  
            img_n = noiseEPOCH_Map  

            img_y = img_x - abs(
                img_n)  

            img_y = torch.clamp(img_y, 0.0, 1.0)  # 钳位到0~1
            img_x, img_n, img_y = img_x.cuda(), img_n.cuda(), img_y.cuda()

            img_result = model(img_y)  
            loss = criterion(img_result, img_x) / 16  
            loss.backward()  
            optimizer.step()  
            model.eval()  
            print('[Epoch:%d][%d/%d] Loss:%.4f' % (epoch + 1, i + 1, 48, loss))
            # Finished One iteration

        #Finished One epoch
        if epoch % DrawStep == 0 and epoch > 1:
            loss_x.append(epoch + 1)
            loss_y.append(loss.item())
            drawFig(loss_x, loss_y, 'tmp')

        torch.save(model.state_dict(), Log_path + Save_Log_name)

    # Finished training
    drawFig(loss_x, loss_y, epoch)


def main_DnCNN():

    model = DnCNN(channels=1, num_of_layers=16)
    # DenseDnCNN
    model = model.cuda()

    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    dataset_train = dataset.Dataset(
        train=True)  #dataset.Dataset 读取了 'train.h5' 文件
    Batch_dataset = DataLoader(dataset=dataset_train,
                               num_workers=4,
                               batch_size=Batch_Size,
                               shuffle=True)

    if create_model:
        model.apply(weights_init_kaiming)
    else:
        model.load_state_dict(torch.load(Log_path + Load_Log_name))

    loss_x = [0]
    loss_y = [0]
    lr = learning_rate

    for epoch in range(total_epoch_num):

        print('\n\n\n')
        if epoch % LrStep == 0 and epoch > 0:
            lr = lr * (LrRaio)
            print('Halved learning rate')
        print('[Epoch:%d] Learning rate=' % (epoch + 1), lr, '\n')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for i in range(48):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            imgnum = random.choice(range(12))
            img_x = cv2.imread(TrainSet_path + '/%d.png' % (imgnum + 1))
            img_x = img_x / 255.
            img_x = np.expand_dims(img_x[:, :, 0], (0, 1))

            noisenum = random.choice(range(160))
            img_n = np.load(TrainNoise_path + '/noiseMap_%d.npy' % noisenum)
            img_GT = img_n**2

            img_y = img_x + img_n

            diff = random.choice([0.05, 0.07, 0.08, 0.09, 0.10, 0.11])  # 调整噪声等级
            for bi in range(512):
                for bj in range(512):
                    if img_y[0, 0, bi, bj] < 0:
                        img_y[0, 0, bi, bj] = diff
                    if img_y[0, 0, bi, bj] > 1:
                        img_y[0, 0, bi, bj] = 1.0 - diff

            img_y = torch.Tensor(img_y)
            img_GT = torch.Tensor(img_GT)

            img_y, img_GT = img_y.cuda(), img_GT.cuda()
            # print('y shape',img_y.shape)
            # kk=input()
            img_result = model(img_y)
            loss = criterion(img_result, img_GT)
            loss.backward()  
            optimizer.step()  
            model.eval()  
            print('[Epoch:%d][%d/%d] Loss:%.4f' % (epoch + 1, i + 1, 48, loss))
        #Finished One Epoch

        if epoch % DrawStep == 0 and epoch > 1:
            loss_x.append(epoch + 1)
            loss_y.append(loss.item())
            drawFig(loss_x, loss_y, 'tmp')
        torch.save(model.state_dict(), Log_path + Save_Log_name)

    # Finished training
    drawFig(loss_x, loss_y, epoch)


def get_Npair():

    imgset = dataset.Dataset(train=True)  #value 0-1,128*128

    for imgnum in range(192):
        mapnum = imgnum // 48
        imgn = np.load(TrainNoise_path + '/noiseMap_%d.npy' %
                       (mapnum))  #value 0-1
        diff = random.choice([10, 15, 20, 25])
        imgx = imgset[imgnum]
        imgy = imgx + imgn[imgnum - 48 * mapnum, 0, 0, :, :]
        imgy1 = np.clip(imgy, 0, 1)  
        for imgi in range(128):  
            for imgj in range(128):
                if imgy[0, imgi, imgj] < 0:
                    imgy[0, imgi, imgj] = diff
                if imgy[0, imgi, imgj] > 0:
                    imgy[0, imgi, imgj] = 1 - diff
                # imgy,diff
        imgindex = str(mapnum) + '-' + str(imgnum - 48 * mapnum)
        np.save(rpath + '/train_sets/noiseExtreme/EX-%s.npy' % (imgindex),
                imgy1)
        np.save(rpath + '/train_sets/noiseDiff/Diff-%s.npy' % (imgindex), imgy)
        np.save(rpath + '/train_sets/noiseFree/Free-%s.npy' % (imgindex), imgx)


if __name__ == '__main__':

    if create_noise:
        List_SPNlevel = [0.05, 0.20, 0.35, 0.50]  

        for epoch in range(160):  # the number of noise maps
            print('Generating noise map %d ...' % epoch)
            SPNlevel = List_SPNlevel[epoch%4]
            noisemap = np.zeros((1, 1, 512, 512))

            for ni in range(512):
                for nj in range(512):
                    value = random.random()
                    if value < SPNlevel / 2:
                        noisemap[0, 0, ni, nj] = -1
                    if value > 1 - SPNlevel / 2:
                        noisemap[0, 0, ni, nj] = 1

            np.save(rpath + 'train_sets/noise-512/noiseMap_%d.npy' % (epoch), noisemap)

        print('Noise Generation Complete!')

    if create_patch:
        dataset.create_patch(data_path=TrainSet_path,
                             patch_size=128)  
        print('Patch generation complete!')

    if create_PairedNoise:
        get_Npair()
        print('Pair generation complete!')

    if train_model:
        if select_model == 'DnCNN':
            main_DnCNN()
        if select_model == 'DPIR':
            main_DPIR()
        print('Training Complete!')
