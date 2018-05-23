import numpy as np
import pickle
import time

"""
bag data loader for MIML
"""

class loader:
    def __init__(self, relation_file, label_data_file, unlabel_data_file = None, group_eval_data_file = None,
                 embed_dir = None, word_dir = None,
                 n_vocab = 80000, valid_split = 1000, max_len = 119, split_seed = 0,
                 use_DS_data = True):
        """
          relation_file: dictionary index of relation
          label_data_file: (list of) filename(s) of label_data pickle
                           we get <valid_split> from the first file for validation
          unlabel_data_file: filename of unlabel_data pickle
          group_eval_data_file: filename of evaluation data, which is in the form of bags of mentions for entities
          embed_dir: if None, then train from scratch, then word_dir must be not None
          word_dir: when embed_dir is not None, just ignore this
          n_vocab: vocab_size, default 80000
          valid_split: number of label data for validation
          max_len: maximum length of sentences
        """
        np.random.seed(split_seed)
        self.n_vocab = n_vocab
        self.n_test = valid_split
        self.max_len = max_len
        # load data
        print('load label data ....')
        ts = time.time()
        # relation index
        with open(relation_file, 'rb') as f:
            self.rel_ind = pickle.load(f)
        self.rel_name = dict()
        self.rel_num = len(self.rel_ind)
        for k,i in self.rel_ind.items():
            self.rel_name[i] = k
        # labeled data
        self.label_data = []
        self.test_data = []
        if not isinstance(label_data_file,list):
            label_data_file = [label_data_file]
        for i, fname in enumerate(label_data_file):
            with open(fname, 'rb') as f:
                text, entity, pos, rel = pickle.load(f)
            curr_data = [(t,e,p,self.rel_ind[r]) for t,e,p,r in zip(text,entity,pos,rel) if len(t)<max_len]
            np.random.shuffle(curr_data)
            if i == 0:
                # split valid data set
                self.test_data = curr_data[:valid_split]
                curr_data = curr_data[valid_split:]
            self.label_data += curr_data
        self.label_data_raw = self.label_data.copy()
        self.test_data_raw = self.test_data.copy()
        print('  -> done! elapsed = {}'.format(time.time()-ts))

        # unlabeled data
        self.unlabel_data = []
        if unlabel_data_file is not None and use_DS_data:
            print('load unlabel data ...')
            ts = time.time()
            if not isinstance(unlabel_data_file,list):
                unlabel_data_file = [unlabel_data_file]
            for fname in unlabel_data_file:
                with open(fname, 'rb') as f:
                    text, entity, pos, rel = pickle.load(f)
                self.unlabel_data += [(t,e,p,self.rel_ind[r]) for t,e,p,r in zip(text,entity,pos,rel) if len(t)<max_len]
            print('  -> done! elapsed = {}'.format(time.time()-ts))

        # evaluation data
        self.eval_data = None
        if group_eval_data_file is not None:
            print('load evaluation data ...')
            ts = time.time()
            self.eval_data = []
            with open(group_eval_data_file,'rb') as f:
                G = pickle.load(f)
            for e, dat in G.items():
                rel = set([self.rel_ind[r] for r in dat[0]])
                mention = [(t, p) for t,p in zip(dat[1], dat[2]) if len(t)<max_len]
                if len(mention) > 0:
                    self.eval_data.append((rel,mention))
            print('  -> done! elapsed = {}'.format(time.time()-ts))

        print('load dictionary and embedding...')
        ts = time.time()
        if embed_dir is None:
            self.embed = None
            with open(word_dir,'rb') as f:
                self.vocab,_ = pickle.load(f)
        else:
            with open(embed_dir,'rb') as f:
                self.embed, self.vocab = pickle.load(f)
            self.embed = self.embed[:n_vocab,:]
            self.embed_dim = self.embed.shape[1]
        self.vocab = self.vocab[:n_vocab]
        self.word_ind = dict(zip(self.vocab, list(range(n_vocab))))
        self.init_extra_word()
        print('  -> done! elapsed = {}'.format(time.time()-ts))

    def init_extra_word(self):
        n = self.n_vocab
        self.n_extra = 3
        self.unk,self.eos,self.start=n,n+1,n+2
        self.pad=self.word_ind['<pad>']
        self.vocab += ['<unk>','<eos>','<start>']

    def group_data(self, raw_data, merge_by_entity = False):
        if len(raw_data) == 0:
            return []
        # data: list of (t, e, p, r)
        # return:
        #    a list [(list of relation id, list of (text, position))]
        if not merge_by_entity:
            # every single instance becomes a group
            data = [([r], [(t, p)]) for t, e, p, r in raw_data]
        else:
            # merge mentions by entity names
            group = dict()
            for t,e,p,r in raw_data:
                if e not in group:
                    group[e] = (set(), [])
                group[e][0].add(r)
                group[e][1].append((t, p))
            data = []
            for e, p in group.items():
                rel = sorted(list(p[0]))
                if rel [0] == 0 and len(rel) > 1:
                    rel = rel[1:]
                mention = p[1]
                if len(mention) > 300:
                    #print('Warninng!!! Super Large Bag!! N = {d}, Rel = {r}'.format(d=len(mention), r = rel))
                    continue
                np.random.shuffle(mention)
                data.append((rel, mention))
        np.random.shuffle(data)
        return data

    def init_data(self, bag_batch, seed = 3137, merge_by_entity = True):
        np.random.seed(seed)
        # group data into bags
        self.label_data = self.group_data(self.label_data_raw, merge_by_entity)
        self.test_data = self.group_data(self.test_data_raw, False) # do not merge entities
        self.train_data = self.group_data(self.unlabel_data, True) + self.label_data
        np.random.shuffle(self.train_data)
        # init params
        self.bag_batch = bag_batch   # number of bags processed per iteration
        self.train_n = len(self.train_data)
        self.test_n = len(self.test_data)
        self.train_batches = (self.train_n + bag_batch - 1) // bag_batch
        self.test_batches = (self.test_n + bag_batch - 1) // bag_batch

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
        return len(self.eval_data)

    def get_bag_info(self, k):
        # return:
        #    positive rel IDs
        dat = self.eval_data[k]
        return dat[0]

    def next_batch(self,data_source = 'train'):
        L = self.max_len
        # get training batch
        if data_source == 'train':
            curr_ptr, data = self.train_ptr, self.train_data
        elif data_source == 'test':
            curr_ptr, data = self.test_ptr, self.test_data
        else: # evaluation
            curr_ptr, data = self.eval_ptr, self.eval_data
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
        else:  # evaluation
            self.eval_ptr += self.bag_batch

        return effective, X, Y, E, length, shapes, mask
