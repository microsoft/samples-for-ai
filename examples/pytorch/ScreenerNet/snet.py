import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim
import torchvision
import collections
import os.path, os, shutil, importlib
import numpy as np

from helpers import AverageMeter, accuracy

def snet_loss(snet_out, cls_err, variables, M, alpha):
    w = None
    for p in variables:
        w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
    l1 = F.l1_loss(w, torch.zeros_like(w))
    loss = torch.pow((1-snet_out),2)
    loss= loss*cls_err
    loss = loss + torch.pow(snet_out,2)*torch.clamp(M-cls_err, min=0)
    res = torch.sum(loss)+alpha*l1
    return res

def multilabel_soft_margin_loss(inp, target):
    inp = torch.sigmoid(inp)
    res = torch.sum(target*torch.log(inp)+(1-target)*torch.log(1-inp), dim=1)
    return res

def validate(val_loader, model, criterion, epoch):
    losses=AverageMeter()
    acc=AverageMeter()

    model.eval()

    for i, (images, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        image_var=torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        y_pred=model(image_var)
        loss=criterion(y_pred, label_var)

        prec1, temp_var = accuracy(y_pred.data, target, topk=(1,1))
        losses.update(loss.data[0], images.size(0))

    print('EPOCH {}|Accuracy:{:.3f} |Loss:{:.3f}'.format(epoch, acc.avg, losses.avg))

def train(dataname, max_epoch):
    if dataname=="mnist":
        modellib = importlib.import_module('snet_mnist')
        net, snet = modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f=nn.CrossEntropyLoss(reduce=False).cuda()
        optimizer_f=optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.0001)
    elif dataname=="voc":
        modellib = importlib.import_module('snet_voc')
        net, snet = modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f = multilabel_soft_margin_loss
        optimizer_f=optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.0025)
    else:
        modellib = importlib.import_module('snet_cifar')
        net, snet=modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f=nn.CrossEntropyLoss(reduce=False).cuda()
        optimizer_f=optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.001)

    trainloader = modellib.trainloader
    net.train()
    snet.train()
    net = net.cuda()
    snet = snet.cuda()

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer_f.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            x_w = snet(inputs).squeeze()
            # print(x_w)

            loss = criterion_f(outputs, labels).squeeze()
            loss_w = torch.mean(loss*x_w)
            loss_w.backward(retain_graph=True)
            optimizer_f.step() # update net

            optimizer_s.zero_grad()
            loss_s = snet_loss(x_w, loss, snet.parameters(), M, alpha)
            loss_s.backward()
            optimizer_s.step()

            # print statistics
            running_loss += loss_w.data[0]
            if i % 200 == 199:    # print every 2000 mini-batches
                if epoch % 100 == 99:
                    print("x_w:{}".format(x_w))
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        if epoch%10000==9999:
            torch.save(net.state_dict(), 'net_checkpoint_{}.pm.{}'.format(dataname,epoch+1))
            torch.save(snet.state_dict(), 'snet_checkpoint_{}.pm.{}'.format(dataname,epoch+1))
        if epoch==0:
            print('first epoch ok')
    torch.save(net.state_dict(),'net_{}.pm'.format(dataname))
    torch.save(snet.state_dict(),'snet_{}.pm'.format(dataname))
    print('Finished Training')

def test(model, loader, dataname):
    model = model.cuda()
    model.eval()

    correct = 0; wrong=0
    count = 0
    if dataname=='voc':
        for i, data in enumerate(loader):
            imgs, labels = data
            inputs = Variable(imgs).cuda()
            preds = model(inputs)
            smax = nn.Softmax()
            smax_out = smax(preds)[0]
            probs = smax_out.data # get probability
            labels = labels[0] # get labels
            if i==0:
                print(probs, labels)
            K = int(np.sum(labels))
            count += np.sum(labels)
            for idx in np.argpartition(probs,-K)[-K:]: # topk index
                if labels[idx]>0:
                    correct+=1
                else:
                    wrong+=1
    elif dataname=='mnist':
        for i, data in enumerate(loader):
            imgs, labels = data
            inputs = Variable(imgs).cuda()
            preds = model(inputs)
            smax = nn.Softmax()
            smax_out = smax(preds)[0].cpu()
            probs = smax_out.data.numpy() # get probability
            label = labels[0] # get labels

            idx = np.argmax(probs)
            if idx == label:
                correct+=1
            else:
                wrong+=1
            count = i

    print('test correct number:{}, wrong prediction:{}, sample number:{}'.format(correct, wrong, count))
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',help="train or test", choices=["train", "test"])
    parser.add_argument('dataname',help="dataset name", choices=["mnist","cifar","voc"])
    parser.add_argument('--modelname', help="saved name", default="model.pm")
    parser.add_argument('--modelpath', help='save dir',  default='.')
    parser.add_argument('--max_epoch', type=int, default=100)
    args = parser.parse_args()

    modelpath = os.path.join(args.modelpath, args.modelname)
    if args.phase=="train":
        print("train on {} for {} epoches".format(args.dataname, args.max_epoch))
        train(args.dataname, args.max_epoch)

    if args.phase=='test':
        print("test model {} on {}".format(modelpath, args.dataname))
        modellib = importlib.import_module('snet_{}'.format(args.dataname))
        testloader = modellib.testloader
        net, _ = modellib.create_net()
        net.load_state_dict(torch.load(modelpath))
        test(net, testloader, args.dataname)

