import numpy as np
import pickle
import time

"""
bag data loader for MIML
"""

class miml_loader():
    def __init__(self, relation_file, train_file, test_file,
                 embed_dir = None,
                 n_vocab = 80000, max_len = 145, max_bag_size = 500):  # max_bag_size = 2500
        """
          relation_file: dictionary index of relation
          train_file, test_file: file name for the training/testing data
          group_eval_data_file: filename of evaluation data, which is in the form of bags of mentions for entities
          embed_dir: if None, then train from scratch, then word_dir must be not None
          word_dir: when embed_dir is not None, just ignore this
          n_vocab: vocab_size, default 80000
          max_len: maximum length of sentences
        """
        self.n_vocab = n_vocab
        self.max_len = max_len

        # load data
        print('load relation index ....')
        ts = time.time()
        # relation index
        with open(relation_file, 'rb') as f:
            self.rel_ind = pickle.load(f)
        assert ('NA' in self.rel_ind)
        assert (self.rel_ind['NA'] == 0)
        self.rel_name = dict()
        self.rel_num = len(self.rel_ind)
        for k, i in self.rel_ind.items():
            self.rel_name[i] = k
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        print('load dictionary and embedding...')
        ts = time.time()
        with open(embed_dir, 'rb') as f:
            self.embed, self.vocab = pickle.load(f)
        self.embed = self.embed[:n_vocab, :]
        self.embed_dim = self.embed.shape[1]
        self.vocab = self.vocab[:n_vocab]
        self.word_ind = dict(zip(self.vocab, list(range(n_vocab))))
        self.init_extra_word()
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        # train and test data
        def load_group_data(filename):
            with open(filename,'rb') as f:
                group = pickle.load(f)
            data = []
            for e, dat in group.items():
                rel = set([self.rel_ind[r] for r in dat[0]])
                mention = [(t, p) for t, p in zip(dat[1], dat[2]) if len(t) < max_len]
                if len(mention) > 0:
                    if (max_bag_size is not None) and (len(mention) > max_bag_size):
                        # split into 3 small bags, max training bag size is 5500
                        np.random.shuffle(mention)
                        n = len(mention)
                        ptr = 0
                        while ptr < n:
                            l = ptr
                            r = ptr + max_bag_size
                            if r >= n:
                                data.append((rel, mention[-max_bag_size:]))
                                break
                            else:
                                data.append((rel, mention[l:r]))
                            ptr = r
                    else:
                        data.append((rel, mention))
            return data

        print('load training data ...')
        ts = time.time()
        self.train_data = load_group_data(train_file)
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        print('load testing data ...')
        ts = time.time()
        self.test_data = load_group_data(test_file)
        print('  -> done! elapsed = {}'.format(time.time() - ts))


    def init_extra_word(self):
        n = self.n_vocab
        self.n_extra = 3
        self.unk,self.eos,self.start=n,n+1,n+2
        self.vocab += ['<unk>','<eos>','<start>']

    def init_data(self, bag_batch, seed = 3137):
        np.random.seed(seed)
        # group data into bags
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)
        # init params
        self.bag_batch = bag_batch   # number of bags processed per iteration
        self.train_n = len(self.train_data)
        self.test_n = len(self.test_data)
        self.train_batches = (self.train_n + bag_batch - 1) // bag_batch

    def ID(self, c):
        if c in self.word_ind:
            return self.word_ind[c]
        return self.unk

    def new_epoch(self):
        np.random.shuffle(self.train_data)
        self.train_ptr = 0
        self.test_ptr = 0
        self.eval_ptr = 0

    def get_bag_n(self):
        return len(self.test_data)

    def get_bag_info(self, k):
        # return:
        #    positive rel IDs
        dat = self.test_data[k]
        return dat[0]

    def next_batch(self, data_source = 'train'):
        L = self.max_len
        # get training batch
        if data_source == 'train':
            curr_ptr, data = self.train_ptr, self.train_data
        elif data_source == 'test':
            curr_ptr, data = self.test_ptr, self.test_data
        else:
            curr_ptr, data = self.eval_ptr, self.test_data

        n = len(data)
        effective = min(self.bag_batch, n - curr_ptr)
        curr_bags = [data[(curr_ptr + i) % n] for i in range(self.bag_batch)]

        batch_size = sum([len(d[1]) for d in curr_bags])
        Y = np.zeros((self.bag_batch, self.rel_num), dtype=np.float32)
        X = np.ones((batch_size, L), dtype=np.int32) * self.eos
        E = np.zeros((batch_size, L), dtype=np.int32)
        length = np.zeros((batch_size, ), dtype=np.int32)
        mask = np.zeros((batch_size, L), dtype=np.float32)
        shapes = np.zeros((self.bag_batch + 1), dtype=np.int32)
        shapes[self.bag_batch] = batch_size
        k = 0

        self.cached_pos = []

        for i, bag in enumerate(curr_bags):
            rel = bag[0]
            for r in rel:
                Y[i][r] = 1
            mention = bag[1]
            shapes[i] = k
            for j in range(len(mention)):
                text, pos = mention[j]
                length[k + j] = len(text) + 1
                mask[k + j, : len(text)] = 1  # ignore the last eos symbol
                for l, c in enumerate(text):
                    X[k + j, l] = self.ID(c)
                X[k + j, len(text)] = self.eos
                E[k + j, pos[0]:pos[1]] = 1
                E[k + j, pos[2]:pos[3]] = 2
                self.cached_pos.append(pos)
            k += len(mention)

        if data_source == 'train':
            self.train_ptr += self.bag_batch
        elif data_source == 'test':
            self.test_ptr += self.bag_batch
        else:
            self.eval_ptr += self.bag_batch

        return effective, X, Y, E, length, shapes, mask
