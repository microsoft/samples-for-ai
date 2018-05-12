import pickle, argparse, os
import tsv2ctf
import numpy as np
# imports required for showing the attention weight heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def preload_vobs(pklname):
    with open(pklname,'rb') as f:
        _, vocabs, _ = pickle.load(f)
    i2w = {v:k for k, v in vocabs.items()}
    return i2w, vocabs
def visualize(pklname, i2w):
    with open(pklname, 'rb') as f:
        # [array(len)] allWei:[(embed, attn)...]
        allQid, allDid, allWei = pickle.load(f)
    for i, ids in enumerate(zip(allQid, allDid)): # different weights
        columns = [i2w.get(wid, '<UNK>') for wid in ids[0]]
        index = [i2w.get(wid,'<UNK>') for wid in ids[1]] 
        for j in range(len(allWei)): # different sequence
            ww = allWei[j][i]
            if len(columns)==ww.shape[1] and len(index)==ww.shape[0]:
                dframe = pd.DataFrame(data=ww, columns=columns, index=index)
            elif len(index)==ww.shape[1] and len(columns)==ww.shape[0]:
                dframe = pd.DataFrame(data=ww, columns=index, index=columns)
            elif len(index)==ww.shape[1] and len(index)==ww.shape[0]:
                dframe = pd.DataFrame(data=ww, columns=index, index=index)
            else:
                dframe = pd.DataFrame(data=ww, index=index, columns=['score'])

            sns.heatmap(dframe)
            plt.show()
            
            a=input('if want to save to csv {y/n}:')
            if a=='y':
                save_name=input('csv name:')
                dframe.to_csv(save_name+'.csv')
    