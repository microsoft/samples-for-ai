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

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchtext import data, datasets

class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        print("building RNN language model...")
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid,nlayers, dropout=dropout)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('tied: nhid and emsize must be equal')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        self.decoder.bias.data.zero_()
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_wikitext_iter(args):
    TEXT = data.Field()
    train_data, val_data, test_data = datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_data, min_freq=10)
    train_iter, val_iter, test_iter = data.BPTTIterator.splits(
                (train_data, val_data, test_data),
                batch_size=args.batch_size, bptt_len=30, repeat=False)
    vocab_size = len(TEXT.vocab)
    return train_iter, val_iter, test_iter, vocab_size, TEXT.vocab

def evaluate(model, val_iter, vocab_size):
    model.eval()
    total_loss = 0

    hidden = model.init_hidden(val_iter.batch_size)
    for b, batch in enumerate(val_iter):
        x, y = batch.text, batch.target
        x = x.to(device)
        y = y.to(device)
        output, hidden = model(x, hidden)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.contiguous().view(-1))
        total_loss += loss.item()
        hidden = repackage_hidden(hidden)
    return total_loss / len(val_iter)

def generate(args):
    _, _, _, vocab_size, vocab= get_wikitext_iter(args)
    model = RNNModel('LSTM', ntoken=vocab_size, ninp=600, nhid=600, nlayers=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)

    word_list = []
    for i in range(args.gen_word_len):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(1).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.fill_(word_idx)
        word = vocab.itos[word_idx]
        word_list.append(word)

    print(' '.join(word_list))

def train(args):
    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, vocab_size, _ = get_wikitext_iter(args)
    print("[TRAIN]:%d\t[VALID]:%d\t[TEST]:%d\t[VOCAB]%d" % (len(train_iter), len(val_iter), len(test_iter), vocab_size))
    print("[!] Instantiating models...")
    model = RNNModel('LSTM', ntoken=vocab_size, ninp=600, nhid=600, nlayers=2, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        model.train()
        total_loss = 0

        hidden = model.init_hidden(train_iter.batch_size)
        for b, batch in enumerate(train_iter):
            x, y = batch.text, batch.target
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            hidden = repackage_hidden(hidden)
            loss = F.cross_entropy(output.view(-1, vocab_size), y.contiguous().view(-1))
            loss.backward()

            # Prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            if b % args.log_interval == 0 and b > 0:
                cur_loss = total_loss / args.log_interval
                print("[Epoch: %d, batch: %d] loss:%5.2f | pp:%5.2f" % (e, b, cur_loss, math.exp(cur_loss)))
                total_loss = 0

        val_loss = evaluate(model, val_iter, vocab_size)
        print("[Epoch: %d] val-loss:%5.2f | val-pp:%5.2f" % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model")
            if not os.path.isdir(args.model_dir):
                os.makedirs(args.model_dir)
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))
            best_val_loss = val_loss
    print("[!] training done")
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    test_loss = evaluate(model, test_iter, vocab_size)
    print("test-loss:%5.2f | test-pp:%5.2f" % (test_loss, math.exp(test_loss)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=0.25, help='max norm of the gradients')
    parser.add_argument('--log_interval', type=int, default=100, help='print log every _')
    parser.add_argument('--model_dir', type=str, default='.save/', help='directory to save the trained weights')
    parser.add_argument('--model_name', type=str, default='lm_best_model.pt', help='the model file name')
    parser.add_argument('--gen_word_len', type=int, default=15, help='word number of generations')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args)
    print("[!] generating...")
    generate(args)