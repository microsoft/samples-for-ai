import pickle
import os


rel_data = './RE/relation2id.txt'
rel_data_pkl = './pkl_data/nyt_orig_rel.pkl'
entity_Data = './RE/entity2id.txt'
entity_data_pkl = './pkl_data/nyt_comb_embed_50d.pkl'

test_Data = './RE/test.txt'
test_data_pkl = './pkl_data/nyt_orig_test.pkl'
train_Data = './RE/train.txt'
train_data_pkl = './pkl_data/nyt_orig_train.pkl',


def rel():
    obj = dict()
    with open(rel_data) as f:
        lines = f.readlines()
        for i in lines:
            s = i.split()
            obj[s[0]] = int(s[1])
        with open(rel_data_pkl, 'wb') as f:
            pickle.dump(obj, f)

def testAndTrain(d1,o1):
    obj = dict()
    with open(d1) as f:
        lines = f.readlines()
        for i in lines:
            s = i.split()
            if s[0]+s[1] not in obj.keys():
                obj[s[0]+s[1]] = []
            obj[s[0]+s[1]].append([s[4],s[2],s[3]])
        with open(o1, 'wb') as f:
            pickle.dump(obj, f)

def main():
    # testAndTrain(train_Data,train_data_pkl)
    # testAndTrain(test_Data,test_data_pkl)


if __name__ == '__main__':
    main()
