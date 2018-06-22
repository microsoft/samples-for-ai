# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(7*7*16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = x.view(-1, 7*7*16)
        x = self.fc1(x)
        x = F.leaky_relu(F.dropout(x, 0.5))
        x = self.fc2(x)
        return x


def test_model_forward():
    x = torch.randn(2, 1, 28, 28)
    model = Model()
    out = model(x)
    print(out.size())
    exit(0)

def get_data_loader(train, batch_size = 1000, shuffle = True):
    return torch.utils.data.DataLoader(
        datasets.MNIST('./', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = batch_size, shuffle = shuffle)

def train(args):
    train_loader = get_data_loader(True,  batch_size = args.batch_size)
    test_loader  = get_data_loader(False, batch_size = args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.6)

    for epoch in range(args.epochs):
        model.train()
        count = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            count += len(data)
            print('Epoch: {} ({}/{})\tLoss: {:.6f}'.format(epoch, count, len(train_loader.dataset), loss.item()), end='\r')

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                test_loss += criterion(output, label).item() 
                preds = output.max(1, keepdim=True)[1]
                correct += preds.eq(label.view_as(preds)).sum().item()

        test_loss /= (len(test_loader.dataset) / args.batch_size)
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate', required=False)
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size', required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
