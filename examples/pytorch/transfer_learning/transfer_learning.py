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
import torch.optim as optim
from torchvision import datasets, transforms, models

# You should run this example on GPU, it could be extremly slow if you try to run it on cpu.
# Please adjust the batch size per your GPU Memory capacity.

def create_resnet():
    # Pretrained resnet model https://arxiv.org/abs/1512.03385
    model = models.resnet34(pretrained = True)
    # Customize full connection layer for CIFAR10 dataset after model weights loaded
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def get_data_loader(train, batch_size = 128, shuffle = True):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10('./', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((200, 200)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size = batch_size, shuffle = shuffle)

def train(args):
    train_loader = get_data_loader(True, batch_size = args.batch_size)
    test_loader = get_data_loader(False, batch_size = args.batch_size, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
