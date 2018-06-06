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
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.leaky_relu(F.dropout(x, 0.5))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def test_model_forward():
    x = torch.randn(2, 1, 28, 28)
    model = Model()
    out = model(x)
    print(out.size())
    exit(0)

def train(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=False,
                      transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.6)

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch: {} ({}/{})\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate', required=False)
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size', required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
