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

def make_conv_block(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    ]

class Model(nn.Module):
    def __init__(self, nc=3, nclass=10):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            *make_conv_block(nc, 32),
            *make_conv_block(32, 64),
            *make_conv_block(64, 128),
            nn.Dropout2d(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.Linear(100, nclass)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 128*4*4))
        return x

def test_model_forward():
    x = torch.randn(2, 3, 32, 32)
    model = Model()
    out = model(x)
    print(out.size())
    exit(0)

def get_data_loader(train, batch_size = 256, shuffle = True):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10('./', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size = batch_size, shuffle = shuffle)

def train(args):
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = get_data_loader(True, batch_size = args.batch_size)
    test_loader = get_data_loader(False, batch_size = args.batch_size, shuffle=False)

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
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
