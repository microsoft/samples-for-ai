import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets

class Model(nn.Module):
    def __init__(self, vocab_size, idim=20, hdim = 50, nlayers = 2, use_attention = False, ndirections = 1):
        super(Model, self).__init__()
        self.embeds = nn.Embedding(vocab_size, idim, padding_idx=1)
        self.gru = nn.GRU(input_size = idim, hidden_size = hdim, num_layers = nlayers, bidirectional = (ndirections == 2), dropout = 0.5)
        self.fc = nn.Sequential(nn.Linear(nlayers * ndirections * hdim, 1), nn.Sigmoid())

        self.use_attention = use_attention
        self.att = nn.Linear(hdim * ndirections + idim, 1)
        self.fc_att = nn.Sequential(nn.Linear(hdim * ndirections, 100), nn.Dropout(0.5), nn.Linear(100, 1), nn.Sigmoid())
    
    def forward(self, inputs):
        x = self.embeds(inputs)
        # out: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.gru(x)
        if self.use_attention:
            out = torch.transpose(out, 0, 1).contiguous()
            att_weights = None
            for i, batch in enumerate(out):
                att_out = self.att(torch.cat((batch,x[:,i,:]), 1))
                if att_weights is None:
                    att_weights = att_out
                else:
                    att_weights = torch.cat((att_weights, att_out), 1)
            att_weights = F.softmax(att_weights, dim=0)
            att_weights = torch.transpose(att_weights, 0, 1).unsqueeze(2)
            att_applied = torch.sum(out * att_weights, 1)
            att_result = self.fc_att(att_applied)
            return att_result
        return self.fc(torch.transpose(hidden, 0, 1).contiguous().view(x.size(1), -1))

def test_forward():
    model = Model(1000, use_attention=True).cuda()
    inputs = torch.LongTensor(np.arange(20).reshape(5,4)).cuda()
    print(inputs)
    out = model(inputs)
    print(out.size())

def replace(matched):
    return " " + matched.group("m") + " "

def tokenize_line_en(line):
    line = re.sub(r"\t", "", line)
    line = re.sub(r"^\s+", "", line)
    line = re.sub(r"\s+$", "", line)
    line = re.sub(r"(?P<m>\W)", replace, line)
    line = re.sub(r"\s+", " ", line)
    #line = re.sub(r'"', ' " ', line)
    #line = re.sub(r".", " . ", line)
    #line = re.sub(r",", " , ", line)
    #line = re.sub(r"@", r" @ ", line)
    return line.split()

def get_imdb_iter(args):
    #TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    TEXT  = data.Field(tokenize=tokenize_line_en, lower=True)
    LABEL = data.Field(unk_token=None, pad_token=None)
    train, test = datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: len(ex.text) <= 400)
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    print('len(train)', len(train))
    print('len(test)', len(test))
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print(LABEL.vocab.stoi)
    print(LABEL.vocab.itos)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, device="cuda:0")
    return train_iter, test_iter, TEXT.vocab


def save_model(args, model, filename):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))

def validate(model, val_iter, criterion):
    model.eval()
    val_loss = 0
    corrects = 0
    num = 0
    with torch.no_grad():
        for batch in iter(val_iter):
            #inputs, label = inputs.to(device), label.to(device)
            output = model(batch.text).squeeze()
            val_loss += criterion(output, batch.label.squeeze().float()).item()
            #preds = output.max(1, keepdim=True)[1]
            preds = torch.ge(output, 0.5).long()
            corrects += preds.eq(batch.label.squeeze()).sum().item()

            num += len(batch)
            print(num, end='\r')
    return val_loss, corrects, num

def show_batch(batch, text_vocab):
    ix = torch.transpose(batch.text, 0, 1)
    for index in ix:
        sentence = [text_vocab.itos[i] for i in index]
        print(sentence)

def show_imdb_batch(batch, text_vocab):
    for item in batch.text:
        print(len(item))
        #sentence = [text_vocab.itos[i] for i in index]
        #print(sentence)
def test_ds(args):
    def get_sentence(ix, vocab):
        return [vocab.itos[i] for i in ix]

    train_iter, val_iter, vocab = get_imdb_iter(args)
    for batch in iter(train_iter):
        print(batch.text.size())
        text = torch.transpose(batch.text, 0, 1)
        for ix in text:
            s = get_sentence(ix, vocab)
            print(s)
        break

def train(args):
    train_iter, val_iter, vocab = get_imdb_iter(args)
    model = Model(len(vocab), use_attention=True).cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_corrects = 0
    nIteration = 0
    numTrain = 0

    for batch in iter(train_iter):
        nIteration += 1
        model.train()
        #label = torch.index_select(torch.eye(5, dtype=torch.long).cuda(), 0, batch.label)
        optimizer.zero_grad()
        output = model(batch.text).squeeze()
        loss = criterion(output, batch.label.squeeze().float())
        loss.backward()
        optimizer.step()

        preds = torch.ge(output, 0.5).long()
        train_corrects = preds.eq(batch.label.squeeze()).sum().item()

        numTrain += len(batch)
        print('Epoch: {:.2f} {} / {} \tLoss: {:.4f} Acc: {:.2f}% ({} / {})'
            .format(numTrain / len(train_iter.dataset), numTrain, len(train_iter.dataset), loss.item(), 
            100. * train_corrects / len(batch), train_corrects, len(batch)), end='\r')

        if nIteration % 200 == 0:
            print('\n')
            val_loss, corrects, num = validate(model, val_iter, criterion)
            val_loss /= (num / args.batch_size)
            print('\nVal Loss: {:.4f}, Val Acc: {}/{} ({:.02f}%)\n'.format(
                val_loss, corrects, num, 100. * corrects / num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models', required=False)

    args, unknown = parser.parse_known_args()

    #test_ds(args)
    train(args)
    #test_forward()
