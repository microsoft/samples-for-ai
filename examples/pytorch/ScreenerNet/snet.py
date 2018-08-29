import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim
import torchvision
import collections
import os.path, os, shutil, importlib
import numpy as np

from tensorboardX import SummaryWriter
from helpers import AverageMeter, accuracy
import argparse
import datetime

ckpt_step = 10
parser = argparse.ArgumentParser()
parser.add_argument('phase',help="train or test", choices=["train", "test"])
parser.add_argument('dataname',help="dataset name", choices=["mnist","cifar","voc"])
parser.add_argument('--output_dir', help='save dir',  default='Outputs')
parser.add_argument('--max_epoch', type=int, default=70)
parser.add_argument('--no_snet', action='store_true')
parser.add_argument('--download', help="switch if you want to download pytorch dataset(only valid for mnist and cifar)", action='store_true')
parser.add_argument('--gpu', help="train on gpu", action='store_true')
parser.add_argument('--use_tensorboard', action='store_true')
parser.add_argument('--no_save', action='store_true', help='do not save anything')
parser.add_argument('--load_ckpt', type=str, help='load checkpoint')
parser.add_argument('--save_step', type=int, default=5)
args = parser.parse_args()

modelpath = os.path.join(args.output_dir, 'log_'+datetime.datetime.now().strftime('%m-%d'))
ckpt_step = args.save_step
if not os.path.exists(modelpath) and not args.no_save:
    os.makedirs(modelpath)

def snet_loss(snet_out, cls_err, variables, M, alpha):
    w = None
    for p in variables:
        w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
    l1 = F.l1_loss(w, torch.zeros_like(w))
    loss = cls_err * torch.pow((1-snet_out),2)
    loss = loss + torch.clamp(M-cls_err, min=0) * torch.pow(snet_out,2)
    res = torch.mean(loss)+alpha*l1
    return res

def multilabel_soft_margin_loss(inp, target):
    bce = nn.MultiLabelSoftMarginLoss(size_average=False)
    res = Variable(torch.zeros(inp.shape[0]))

    for i,row in enumerate(inp):
        #print(target[i], row)
        res[i]=bce(row, target[i])
    if args.gpu:
        res = res.cuda()
    return res

def validate(val_loader, model, criterion, epoch):
    losses=AverageMeter()
    acc=AverageMeter()

    model.eval()

    val_iter = iter(val_loader)
    for i, (images, target) in enumerate(val_iter):
        target = target.cuda(async=True)
        image_var=torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        y_pred=model(image_var)
        loss=criterion(y_pred, label_var)

        prec1, temp_var = accuracy(y_pred.data, target, topk=(1,1))
        losses.update(loss.data[0], images.size(0))

    print('EPOCH {}|Accuracy:{:.3f} |Loss:{:.3f}'.format(epoch, acc.avg, losses.avg))

def train(dataname, max_epoch, no_snet, output_dir=None,use_tb=False, no_save=False, download=False, use_gpu=False):
    ckpt_path = os.path.join(output_dir, 'ckpt')
    if dataname=="mnist":
        modellib = importlib.import_module('snet_mnist')
        net, snet = modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f=nn.CrossEntropyLoss(reduce=False)
        optimizer_f=optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.0001)
    elif dataname=="voc":
        modellib = importlib.import_module('snet_voc')
        net, snet = modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f = multilabel_soft_margin_loss
        optimizer_f=optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.0025)
    else:
        modellib = importlib.import_module('snet_cifar')
        net, snet=modellib.create_net()
        M = modellib.M; alpha = modellib.alpha
        criterion_f=nn.CrossEntropyLoss(reduce=False)
        optimizer_f=optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
        optimizer_s=optim.Adam(snet.parameters(), lr=0.001) # 0.001

    trainloader = modellib.getLoader('train', download)
    testloader = modellib.getLoader('test', download)
    net.train()
    snet.train()
    if use_gpu==True:
        net = net.cuda()
        snet = snet.cuda()
    if use_tb:
        writer = SummaryWriter(os.path.join(output_dir,'tb_log'))
    else:
        writer = None

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        trainiter = iter(trainloader)
        for i, data in enumerate(trainiter):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if use_gpu==True:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer_f.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion_f(outputs, labels).squeeze()
            if not no_snet:
                x_w = snet(inputs).squeeze()
                xw_for_net = x_w.detach()
                # print(xw_for_net.device, loss.device)
                loss_w = torch.mean(loss*xw_for_net)
                loss_w.backward()
                optimizer_f.step() # update net

                optimizer_s.zero_grad()
                loss_s = snet_loss(x_w, loss.detach(), snet.parameters(), M, alpha)
                loss_s.backward()
                optimizer_s.step()

                # print statistics
                if i % 100 == 0:    # print every 2000 mini-batches
                    print('[epoch:%d, minibatch:%5d] main_loss: %.4f, true_loss: %.4f, s_loss: %.4f' %\
                        (epoch, i, loss_w, torch.mean(loss), loss_s))
            else:
                loss = torch.mean(loss)
                loss.backward()
                optimizer_f.step()

                if i % 100 == 0:    # print every 2000 mini-batches
                    print('[epoch:%d, minibatch:%5d] loss: %.4f' %(epoch, i, loss))

        if not no_save:
            metric = test(net, testloader, dataname, use_gpu)
            if writer:
                if not no_snet:
                    writer.add_scalar('true_loss', torch.mean(loss_w), epoch)
                    writer.add_scalar('snet_loss', loss_s, epoch)
                else:
                    writer.add_scalar('main_loss', loss, epoch)
                writer.add_scalar('metric', metric, epoch)
            if epoch%ckpt_step==0:
                torch.save(net.state_dict(), '{}net_epoch{}.pm'.format(ckpt_path, epoch))
                if not no_snet:
                    torch.save(snet.state_dict(), '{}snet_epoch{}.pm'.format(ckpt_path,epoch))

    if not no_save:
        torch.save(net.state_dict(),'{}net_epoch{}.pm'.format(ckpt_path, max_epoch))
        if not no_snet:
            torch.save(snet.state_dict(),'{}snet_epoch{}.pm'.format(ckpt_path, max_epoch))
    print('Finished Training')

def test(model, loader, dataname, use_gpu=False):
    if use_gpu:
        model = model.cuda()
    model.eval()

    if dataname=='voc':
        img_scores = []
        img_gt = []
        for i,data in enumerate(loader):
            imgs, labels = data
            inputs = Variable(imgs)
            if use_gpu==True:
                inputs = inputs.cuda()
            preds = model(inputs)
            smax = nn.Softmax()
            smax_out = smax(preds)[0].cpu()
            probs = smax_out.data.numpy() # get probability
            labels = labels[0].numpy() # get labels
            if i==0:
                print(probs, labels)
            img_scores.append(probs.copy())
            img_gt.append(labels.copy())

        img_scores=np.array(img_scores)
        img_gt = np.array(img_gt)
        mAP = [0.0]*20
        for i in range(20): # 20 classes
            scores = img_scores[:,i]
            labels = img_gt[:,i]
            sorted_args = np.argsort(-scores) # from large to small
            # scores = scores[sorted_args]
            labels = labels[sorted_args]

            correct=0; wrong=0
            precise = []
            K=int(np.sum(labels))
            if K==0:
                continue
            recall_threshold = [i/K for i in range(1,K+1)]
            for j,label in enumerate(labels):
                if label>0:
                    correct+=1
                else:
                    wrong+=1
                rec = correct/K; precision = correct/(correct+wrong)
                if rec<=recall_threshold[0]:
                    if len(precise)==0:
                        precise.append(precision)
                    else:
                        precise[-1]=max(precision, precise[-1])
                else:
                    recall_threshold.pop(0)
                    precise.append(precision)
                if len(recall_threshold)==0:
                    break
            mAP[i] = np.mean(np.array(precise))

        mmAP = np.mean(np.array(mAP))
        print('mAP:{}\n mAP:{}'.format(mAP,mmAP))
        return mmAP
    elif dataname in ['mnist', 'cifar']:
        correct = 0; wrong=0
        for i, data in enumerate(loader):
            imgs, labels = data
            inputs = Variable(imgs)
            if use_gpu==True:
                inputs = inputs.cuda()
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
        print('test correct number:{}, wrong prediction:{}, sample number:{}'.format(correct, wrong, len(loader)))
        return correct/len(loader)
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


if __name__=='__main__':


    if args.phase=="train":
        print("train on {} for {} epoches".format(args.dataname, args.max_epoch))
        train(args.dataname, args.max_epoch, args.no_snet, modelpath, args.use_tensorboard, args.no_save, \
              download=args.download, use_gpu=args.gpu)

    if args.phase=='test':
        print("test model {} on {}".format(modelpath, args.dataname))
        modellib = importlib.import_module('snet_{}'.format(args.dataname))
        testloader = modellib.getLoader('test', args.download)
        net, _ = modellib.create_net()
        net.load_state_dict(torch.load(args.load_ckpt))
        test(net, testloader, args.dataname, use_gpu=args.gpu)
