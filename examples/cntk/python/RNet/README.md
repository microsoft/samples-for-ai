# Instructions for getting started with bidaf 

[简体中文](/zh-hans/examples/cntk/python/RNet/README.md)

## Download the MS MARCO  dataset
```
wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
```
## Download GloVe vectors
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
## Download NLTK punkt
```
python3 -m nltk.downloader -d $HOME/nltk_data punkt
```
or
```
python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
```

## Run `convert_msmarco.py`
It should create `train.tsv`, `dev.tsv`, and `test.tsv`, and only keep querys that query_type is "description". 

It does a fair amount of preprocessing therefore converting data to cntk text format reader starts from these files

## Run `tsv2ctf.py`
It creates a `vocabs.pkl` and `train.ctf`, `val.ctf`, and `dev.ctf`

The data is ready now. 

## Run 'rnetmodel.py' in GPU

you can see the model structure from here

## Run 'train_rnet.py' in GPU

this train code mainly from here https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow