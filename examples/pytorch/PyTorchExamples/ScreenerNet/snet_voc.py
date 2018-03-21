import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from collections import OrderedDict
import torchvision.transforms as transforms

import numpy as np
import os

# =========== dataset ====================
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
class PascalVoc(torch.utils.data.Dataset):
    class_names = np.array([
        'aeroplane','bicycle', 'bird','boat','bottle','bus',
        'car', 'cat','chair','cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'potted plant',
        'sheep', 'sofa', 'train', 'tvmonitor'])
    mean_bgr = np.array([104.00698793, 116.66876762, 112.67891434])

    def __init__(self, root, image_set, train, transform=None, target_transform=None):
        super(PascalVoc, self).__init__()
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(self.root,'Annotations','%s.xml')
        self._imgpath = os.path.join(self.root,'JPEGImages','%s.jpg')
        self._imgsetpath = os.path.join(self.root,'ImageSets','Main','%s.txt')

        self.kind = 'train' if train else 'val'
        with open(self._imgsetpath % self.kind) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]
    def __getitem__(self, index):
        img_id = self.ids[index]
        root = ET.parse(self._annopath % img_id).getroot()
        cls_name = root.findall('./object/name')
        target = np.zeros_like(self.class_names, dtype=np.float32)
        for name in cls_name:
            tmp = name.text
            target[np.argwhere(PascalVoc.class_names==tmp)]=1
        img = Image.open(self._imgpath%img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

# =============== net ===================
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        net = torchvision.models.vgg19_bn(pretrained=True)
        for p in net.parameters():
            p.requires_grad = False
        self.net = net
        origin_cls = self.net.classifier
        self.net.classifier=nn.Sequential(
                origin_cls[0],nn.Dropout(),
                nn.Linear(4096, 128),
                nn.BatchNorm1d(128),nn.ReLU(),
                nn.Dropout(), nn.Linear(128, 20))
    def forward(self, x):
        x = self.net(x)
        return x
    def parameters(self):
        res = (p for p in self.net.parameters() if p.requires_grad)
        return res
class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        net = torchvision.models.vgg19_bn(pretrained=True)
        for p in net.parameters():
            p.requires_grad = False

        self.net = net
        origin_cls = self.net.classifier
        self.net.classifier=nn.Sequential(
                origin_cls[0],
                nn.Linear(4096, 64),
                nn.BatchNorm1d(64),nn.ReLU(),
                nn.Dropout(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.net(x)
        x = F.sigmoid(x)
        return x
    def parameters(self):
        res = (p for p in self.net.parameters() if p.requires_grad)
        return res

# =========== config ===============
def collnt_func(batch): # 需要resize以后，默认的方法可以将其直接拼接
    imgs = []
    labels = []
    for v, l in batch:
        imgs.append(imgs); labels.append(l)
        print(v)
    #return imgs, labels # np.array
    return torch.FloatTensor(np.asarray(imgs)), torch.IntTensor(np.asarray(labels))

M=1.0; alpha=0.001
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(75),
    transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std= [0.229, 0.224, 0.225])])
trainset = PascalVoc('./VOC2012','2012',True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
        shuffle=True, num_workers=4) # , collate_fn=collnt_func)

testset = PascalVoc('./VOC2012','2012', False, transform=
        transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        )
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
def create_net():
    bnet = BaseNet()
    snet = SNet()
    return bnet, snet
if __name__=="__main__":
    for data in enumerate(testloader):
        pass
