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
import re
import os
import argparse
import time
import sys
import torch
import random
import tarfile
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sys import version_info
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        print(path)
        data = torch.load(path)
        self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None):
        prefix = 'snapshot/' + self.model_name + '_' +self.opt.type_+'_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name
        data=self.state_dict()

        torch.save(data, path)
        return path

class DynamicLSTM(BasicModule):
    def __init__(self, input_dim, output_dim,
                 num_layers=1, bidirectional=True,
                 batch_first=True):
        super().__init__()
        self.embed_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=self.batch_first)

    def forward(self, inputs, lengths):
        # sort data by lengths
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        sort_embed_input = inputs.index_select(0, Variable(idx_sort))
        sort_lengths = list(lengths[idx_sort])

        # pack
        inputs_packed = nn.utils.rnn.pack_padded_sequence(sort_embed_input,
                                                          sort_lengths,
                                                          batch_first=True)
        # process using RNN
        out_pack, (ht, ct) = self.lstm(inputs_packed)

        # unpack: out
        out = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
        out = out[0]

        # unsort: h
        ht = torch.transpose(ht, 0, 1)[idx_unsort]
        ht = torch.transpose(ht, 0, 1)

        out = out[idx_unsort]
        ct = torch.transpose(ct, 0, 1)[idx_unsort]
        ct = torch.transpose(ct, 0, 1)

        return out, (ht, ct)

class SelfAttention(BasicModule):
    def __init__(self, input_hidden_dim):
        super().__init__()
        self.hidden_dim = input_hidden_dim
        self.fc = nn.Linear(self.hidden_dim, 1)
    def forward(self, encode_output):
        # (B, L, H) -> (B, L, 1)
        energy = self.fc(encode_output)
        weights = F.softmax(energy.squeeze(-1), dim=1)

        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encode_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class Highway(BasicModule):
    def __init__(self, input_dim, activate_funcation = F.relu):
        super(Highway, self).__init__()
        self.nonlinear = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.activate_function = activate_funcation

    def forward(self, x):
        """
        :param x: (B, H)
        :return: (B, H)
        """
        # (B, H) -> (B, H)
        T = torch.sigmoid(self.gate(x))
        # (B, H) -> (B, H)
        H = self.activate_function(self.nonlinear(x))
        # output = T(x, W_T) * H(x, W_H) + (1 - T(x, W_t)) * x
        output = T * H + (1 - T) * x
        return output

class LSTMSelfAttentionHighway(BasicModule):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.lstm_hidden_dim
        self.class_num = args.class_num
        self.num_layers = args.lstm_num_layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        self.lstm = DynamicLSTM(input_dim=self.embed_dim,
                            output_dim=self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention = SelfAttention(input_hidden_dim=2*self.hidden_dim)
        self.highway = Highway(2*self.hidden_dim)
        self.fc = nn.Linear(2*self.hidden_dim, self.class_num)

    def forward(self, inputs, lengths):
        """
        :param inputs: (B, L)
        :param lengths: (B, 1)
        :return:(B, C)
        """
        # (B, L, E)
        embed_input = self.embedding(inputs)
        # (B, L, E) -> (B, L, 2H)
        output, (ht, ct) = self.lstm(embed_input, lengths)

        # (B, L, 2H) -> (B, 2H), (B, L, 1)
        weighted_out, weights = self.attention(output)
        highway_out = self.highway(weighted_out)
        logits = self.fc(highway_out)

        return logits

class TextCNN(BasicModule):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.class_num = args.class_num
        self.output_dim = args.filters
        self.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([nn.Conv2d(1, self.output_dim, (K, self.embed_dim))
                                    for K in self.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear_layer = nn.Linear(len(self.kernel_sizes) * self.output_dim, self.class_num)

    def forward(self, x, inputs_length):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.linear_layer(x)
        return logit

def get_file_name(file_dir, name="dirs"):
    res_dirs = list()
    res_files = list()
    for root, dirs, files in os.walk(file_dir):
        res_dirs += dirs
        res_files += files

    if name == "dirs":
        return res_dirs
    else:
        return res_files

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

def get_dataset_iter(args):
    print("Loading data...")
    TEXT = data.Field(lower=True, tokenize=tokenize_line_en, include_lengths=True, batch_first=True, sequential=True)
    LABEL = data.Field(sequential=False)
    train, test = NewsGroup.splits(TEXT, LABEL)

    print("Building vocabulary...")
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits((train, test), sort_key = lambda x:len(x.text),
                                                       sort_within_batch=True,
                                                       batch_size=args.batch_size, device=-1,
                                                       repeat = False)
    args.vocab_size = len(TEXT.vocab)
    args.class_num = len(LABEL.vocab) - 1
    print("Loading data finish...")
    return train_iter, test_iter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class BasicDataset(data.Dataset):

    @classmethod
    def download_or_unzip(cls, root):
        if not os.path.exists(root):
            os.mkdir(root)
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urldownload.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='./data', **kwargs):
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


class NewsGroup(BasicDataset):

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
    url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"
    filename = 'mini_newsgroups.tar.gz'
    dirname = 'mini_newsgroups'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        path = self.dirname if path is None else path

        if examples is None:
            examples = []
            class_dirs = get_file_name(path)
            for class_dir_name in class_dirs:
                class_dir_path = os.path.join(path, class_dir_name)
                file_names = get_file_name(class_dir_path, "files")
                for file in file_names:
                    file_path = os.path.join(class_dir_path, file)
                    try:
                        with open(file_path) as f:
                            raw_data = f.read()
                            if len(raw_data.split(' ')) > 100:
                                raw_data = ' '.join(raw_data.split(' ')[0:100])
                            examples += [data.Example.fromlist([raw_data, class_dir_name], fields)]
                    except:
                        continue
        super(NewsGroup, self).__init__(examples, fields, **kwargs)

def validate(model, val_iter, args):
    model.eval()
    corrects, avg_loss = 0.0, 0.0
    for batch in val_iter:

        (inputs, inputs_length), target = batch.text, batch.label - 1

        if args.cuda and args.device != -1:
            inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()

        logit = model(inputs, inputs_length)
        loss = F.cross_entropy(logit, target)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        avg_loss += loss.item()
        corrects += correct

    size = len(val_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

def train(model, train_iter, val_iter, args):
    print("begin to train models...")
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    steps = 0
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            (inputs, inputs_length), target = batch.text, batch.label - 1
            if args.cuda and args.device != -1:
                inputs, inputs_length, target = inputs.cuda(), inputs_length.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(inputs, inputs_length)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

                accuracy = 100*corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            if steps % args.test_interval == 0:
                validate(model, val_iter, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--model_name', type=str, default='LSTMSelfAttentionHighway', help='the model', required=False)

    # common args
    parser.add_argument('--cuda', type=bool, default=True, help='enable the cuda or not', required=False)
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-log-interval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    # models args
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=256, help='Number of training epochs', required=False)
    parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden size', required=False)
    parser.add_argument('--embed_dim', type=int, default=128, help='the embedding dim of word embedding', required=False)


    # CNN args
    parser.add_argument('--filters', type=int, default=100, help='Number of cnn filters', required=False)
    parser.add_argument('-kernel_sizes', type=str, default='1,2,3,4', help='different kernel size of cnn')
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout rate')

    # LSTM args
    parser.add_argument('--lstm_hidden_dim', type=int, default=100, help='Number of lstm hidden dim', required=False)
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of lstm layer numbers', required=False)
    args, unknown = parser.parse_known_args()

    print("args : " + str(args))
    print("unknown args : " + str(unknown))

    train_iter, val_iter = get_dataset_iter(args)

    model = eval(args.model_name)(args)
    torch.cuda.set_device(args.device)
    if args.cuda and args.device != -1:
        model = model.cuda()

    train(model, train_iter, val_iter, args)