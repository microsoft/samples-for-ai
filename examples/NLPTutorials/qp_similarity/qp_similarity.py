# ====================================================================================================== #
# The MIT License (MIT)
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ====================================================================================================== #

from __future__ import print_function
import os
import re
import argparse
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torchtext.data import BucketIterator, Field
from torchtext.data.dataset import TabularDataset
from sys import version_info
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload


def replace(matched):
    return " " + matched.group("m") + " "

def tokenize_line_en(line):
    line = re.sub(r"\t", "", line)
    line = re.sub(r"^\s+", "", line)
    line = re.sub(r"\s+$", "", line)
    line = re.sub(r"<br />", "", line)
    line = re.sub(r"(?P<m>\W)", replace, line)
    line = re.sub(r"\s+", " ", line)
    return line.split()


class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(vocab.vectors)
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(p=0.3)
        self.hidden_dim = 200
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2*self.hidden_dim, 2400),
            nn.Tanh(),
            nn.Linear(2400, 2)
        )

    def forward(self, batch):
        batch_size = batch.question.size(0)
        question = self.embedding(batch.question)
        sentence = self.embedding(batch.sentence)
        question = self.dropout(question)
        sentence = self.dropout(sentence)
        _, hidden_q = self.gru(question)
        _, hidden_s = self.gru(sentence)
        hidden_q = hidden_q.transpose(0, 1).contiguous().view(batch_size, -1)
        hidden_s = hidden_s.transpose(0, 1).contiguous().view(batch_size, -1)
        concat_input = torch.cat([hidden_q, hidden_s], dim=-1)
        scores = self.classifier(concat_input)
        return scores


def get_dataset(args):
    TEXT = Field(sequential=True, tokenize=tokenize_line_en, lower=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
    train, val, test = TabularDataset.splits(path='WikiQACorpus', root='', train='WikiQA-train.tsv',
                                             validation='WikiQA-dev.tsv', test='WikiQA-test.tsv', format='tsv',
                                             fields=[('question_id', None), ('question', TEXT),
                                                     ('document_id', None), ('document_title', None),
                                                     ('sentence_id', None), ('sentence', TEXT),
                                                     ('label', LABEL)],
                                             skip_header=True)
    TEXT.build_vocab(train, vectors='glove.840B.300d')
    train_iter, dev_iter, test_iter = BucketIterator.splits((train, val, test), batch_size=args.batch_size,
                                                                  sort=False, shuffle=True, repeat=False,
                                                            device='cuda' if torch.cuda.is_available() else None)
    return train_iter, dev_iter, test_iter, TEXT.vocab


def save_model(args, model):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))


def validate(model, val_iter, criterion):
    model.eval()
    loss = 0
    batch_num = 0
    ref_list = []
    score_list = []
    with torch.no_grad():
        for batch in iter(val_iter):
            y = batch.label
            y_pred = model(batch)
            loss += criterion(y_pred, y).item()
            y_score = F.softmax(y_pred, dim=-1)[:, 1]
            batch_num += 1
            ref_list += y.cpu().numpy().tolist()
            score_list += y_score.cpu().numpy().tolist()
    loss /= batch_num

    fpr, tpr, thresholds = metrics.roc_curve(ref_list, score_list)
    auc = metrics.auc(fpr, tpr)

    return loss, auc


def train(args):
    train_iter, dev_iter, test_iter, vocab = get_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(vocab).to(device)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_auc = 0
    for epoch in range(args.epochs):
        for batch in iter(train_iter):
            model.train()
            optimizer.zero_grad()
            y = batch.label
            y_pred = model(batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print('Epoch: {:d}\tLoss: {:.4f}'.format(epoch, loss.item()), end='\r')

        val_loss, val_auc = validate(model, dev_iter, criterion)
        print('Epoch: {:d}\tVal Loss: {:.4f}\tAUC: {:.4f}\n'.format(epoch, val_loss, val_auc))

        if val_auc > best_val_auc:
            save_model(args, model)
            best_val_auc = val_auc

    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    test_loss, test_auc = validate(model, test_iter, criterion)
    print('\nTest Loss: {:.4f}\tAUC: {:.4f}\n'.format(test_loss, test_auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Name of best model')
    args = parser.parse_args()

    if not os.path.exists("WikiQACorpus"):
        data_url = "https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"
        urldownload.urlretrieve(data_url, "WikiQACorpus.zip")
        zf = zipfile.ZipFile("WikiQACorpus.zip")
        zf.extractall()
        zf.close()

    train(args)

