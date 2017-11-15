from __future__ import print_function
import cifar_utils as ut
import os

def loadData(datadir):
    trn, tst= ut.loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', datadir)
    print ('Writing train text file...')
    ut.saveTxt(os.path.join(datadir, r'Train_cntk_text.txt'), trn)
    print ('Done.')
    print ('Writing test text file...')
    ut.saveTxt(os.path.join(datadir, r'Test_cntk_text.txt'), tst)
    print ('Done.')

    print ('Converting train data to png images...')
    ut.saveTrainImages(r'Train_cntk_text.txt', os.path.join(datadir, 'train'), datadir)
    print ('Done.')
    print ('Converting test data to png images...')
    ut.saveTestImages(r'Test_cntk_text.txt', os.path.join(datadir, 'test'), datadir)
    print ('Done.')
