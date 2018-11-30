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

'''
Running environment: Python 3 + Pytorch 0.4, CPU/GPU
'''

import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import codecs

from torchtext import data
from torchtext import datasets


class ATISDataset(data.Dataset):
    urls = ['https://raw.githubusercontent.com/yvchen/JointSLU/master/data/atis.train.iob',
        'https://raw.githubusercontent.com/yvchen/JointSLU/master/data/atis.test.iob']
    dirname = ''
    name = 'atis'

    @staticmethod
    def sort_key(example):
        return len(example.labels)

    def __init__(self, path, fields, separator="\t", **kwargs):
        examples = []
        with codecs.open(path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                line = line.strip()
                if len(line) != 0:
                    words, labels = line.split(separator)
                    columns = []
                    columns.append(words.split(' '))
                    columns.append([label for label in labels.split(' ') if label])
                    examples.append(data.Example.fromlist(columns, fields))
        super(ATISDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root='.data', train='atis.train.iob', test='atis.test.iob', validation_frac=0.2, **kwargs):
        train, test = super(ATISDataset, cls).splits(fields=fields, root=root, train=train, test=test, separator='\t', **kwargs)
        # HACK: Saving the sort key function as the split() call removes it
        sort_key = train.sort_key

        # Now split the train set
        # Force a random seed to make the split deterministic
        random.seed(0)
        train, val = train.split(1 - validation_frac, random_state=random.getstate())
        # Reset the seed
        random.seed()

        # HACK: Set the sort key
        train.sort_key = sort_key
        val.sort_key = sort_key

        return train, val, test


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes, batch_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, input):
        input_embedded = self.embedding(input)
        rnn_out, (hn, cn) = self.rnn(input_embedded)
        rnn_out = self.linear(rnn_out.permute(1, 0, 2))
        prob = F.log_softmax(rnn_out, dim=2)
        return prob


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


def get_atis_iter(args):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    TEXT = data.Field(lower=True, tokenize=tokenize_line_en)
    LABELS = data.Field(batch_first=True)
    train, val, test = ATISDataset.splits(fields=(('text', TEXT), ('labels', LABELS)), root=args.data_dir)
    TEXT.build_vocab(train.text)
    LABELS.build_vocab(train.labels)
    print('Number of train dataset:', len(train))
    print('Number of validation dataset:', len(test))
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=args.batch_size, device='cuda' if torch.cuda.is_available() else None)
    return train_iter, val_iter, test_iter, TEXT.vocab, LABELS.vocab


def save_model(args, model):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))


def validate(model, val_iter, n_classes, criterion=None):
    model.eval()
    val_loss = 0
    cnt_true = 0
    cnt_false = 0
    with torch.no_grad():
        for batch in iter(val_iter):
            probs = model(batch.text)
            if criterion:
                val_loss += criterion(probs.view(-1, n_classes), batch.labels.view(-1))
            preds = torch.max(probs, 2)[1].squeeze(1).cpu().data.numpy()
            answers = batch.labels.cpu().data.numpy()

            for pred, answer in zip(preds, answers):
                for label_pred, label_answer in zip(pred, answer):
                    if label_pred == label_answer:
                        cnt_true += 1
                    else:
                        cnt_false += 1
    accuracy = cnt_true * 1.0 / (cnt_true + cnt_false)
    print("The label accuracy is: %f" % accuracy)
    return val_loss, accuracy


def test(args):
    _, _, test_iter, text_vocab, label_vocab = get_atis_iter(args)
    vocab_size = len(text_vocab)
    n_classes = len(label_vocab)
    model = Model(vocab_size, 64, n_classes, args.batch_size)
    if torch.cuda.is_available():
        model = model.cuda()
    try:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    except:
        print('Error loading model file, please train the model and make sure model file({}) exists.'.format(os.path.join(args.model_dir, args.model_name)))

    validate(model, test_iter, n_classes)


def train(args):
    train_iter, val_iter, _, text_vocab, label_vocab = get_atis_iter(args)
    vocab_size = len(text_vocab)
    n_classes = len(label_vocab)
    model = Model(vocab_size, 64, n_classes, args.batch_size)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    max_acc = 0

    model.train()
    # get batch data
    train_pred_label = []
    n_iter = 0
    for batch in iter(train_iter):
        # zero_grad
        optimizer.zero_grad()
        #forward
        probs = model(batch.text)
        train_pred_label.extend(torch.max(probs, 2)[1].squeeze(1).cpu().data.numpy())
        # compute loss
        loss = criterion(probs.view(-1, n_classes), batch.labels.view(-1))
        print_loss = loss.item()
        # backward
        loss.backward()
        optimizer.step()
        n_iter += 1

        print('Batch idx: (%d / %d) loss: %.6f' % (n_iter, args.max_iter, print_loss/len(batch.text)))
        train_pred = [list(map(lambda x: label_vocab.itos[x], y)) for y in train_pred_label]
        
        if n_iter % 100 == 0:
            val_loss, accuracy = validate(model, val_iter, n_classes, criterion)
            if accuracy > max_acc:
                max_acc = accuracy
                save_model(args, model)
            model.train()

        if n_iter > args.max_iter:
            print('Training finished.')
            break
    val_loss, accuracy = validate(model, val_iter, n_classes, criterion)
    if accuracy > max_acc:
        max_acc = accuracy
        save_model(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size", required=False)
    parser.add_argument('--max_iter', type=int, default=5000, help='Number of maxmum training batches', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models', required=False)
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Directory to save models', required=False)

    args, unknown = parser.parse_known_args()

    print('-' * 30 + 'train' + '-' * 30)
    train(args)
    print('-' * 30 + 'test' + '-' * 30)
    test(args)
