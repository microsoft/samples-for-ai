import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

# =========== configuration ==============
M = 1.0
alpha = 0.01
trainset = torchvision.datasets.MNIST(root='.', train=True,
        transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
        shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='.', train=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
def create_net():
    net, snet = BaseNet(),Screener()
    return net, snet
# =========== Module ====================
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) # 28-5+1=24->12
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5) # 12-5+1=8->4
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Screener(nn.Module):
    def __init__(self):
        super(Screener, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(1,4,3)),('act1',nn.ELU()), # 28-3+1=26
            ('conv2',nn.Conv2d(4,8,3)),('act2',nn.ELU()), # 26-3+1=24
            ('conv3',nn.Conv2d(8,16,3)),('act3',nn.ELU()), # 24-3+1=22
            ('conv4',nn.Conv2d(16,32,3)),('act4',nn.ELU())])) # 22-3+1=20
        self.fc = nn.Linear(20*20*32,1)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 20*20*32)
        out = F.sigmoid(self.fc(x))
        return out

