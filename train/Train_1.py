
import matplotlib.pyplot as plt
import torch
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as func
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from models_16 import Net_16
import random

def drawif(x,y):
    plt.plot(x,np.log10(y))
    x.lable()
    y.lable()


num_epoch = 450
batch_size = 1
learning_rate = 1e-4

data_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
#    transforms.Normalize([0.5], [0.5])
    ])

dst_train1 = torchvision.datasets.ImageFolder(root='train', transform=data_tf)

loader_train1 = DataLoader(dst_train1, batch_size=batch_size, shuffle=False)

model = Net_16().cuda()
#model.load_state_dict(torch.load('Noise_map_3.pth'))

def noise(image, p):
    arr = []
    output = np.zeros(shape=(512, 512))
    output1 = np.zeros(shape=(512, 512))
    for i in range(512):
        for j in range(512):
            r = random.random()
            if r < p:
                output[i][j] = random.uniform(0.8, 1)
                output1[i][j] = 1
            else:
                if r > 1 - p:
                    output[i][j] = random.uniform(0, 0.2)
                    output1[i][j] = 1
                else:
                    output[i][j] = image[i][j]
    arr.append(output)
    arr.append(output1)
    return arr

times = []
loss_list = []

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss(size_average=False)

for epoch in range(num_epoch):
    model.train()
    for  batch_idx, (img, lab) in enumerate(loader_train1):
        p = random.uniform(0,0.45)

        img = torch.Tensor(img).reshape(512, 512)

        add_noise = noise(img, p)

        img_noise = add_noise[0]

        noise_map = add_noise[1]

        x = torch.Tensor(img_noise).reshape(1, 1, 512, 512).cuda()

        y = torch.Tensor(img).reshape(1, 1, 512, 512).cuda()

        z = torch.autograd.Variable(x).cuda()

        out = model(z)

        loss = criterion(out, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        printable_loss = loss.data.item()

    print(epoch)
    print(printable_loss)
    times.append(epoch)
    loss_list.append(printable_loss)
    plt.title('Loss Figure versus Epoch')
    plt.plot(times, np.log10(loss_list), 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log10)')
plt.show()
plt.savefig('LFVE.png')
torch.save(model.state_dict(), 'Noise_map_10.pth')
