# What are contained
* BiDAF baseline
* BiElmo implement of https://arxiv.org/abs/1802.05365
* BiFeature: add artificial features such as word concurrence, minimal edit distance and jaccard similarity between question and documents
* RNetFeature
* RNetElmo
* BiSAF{1,2}: use multihead self attention mechanism

# Instructions for getting started with QA models
## Download the MS MARCO  dataset
```
wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
```
## Download GloVe vectors
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```
## Generate char-glove-300 for RNet
```
python get_char_glove.py
```
## Get elmo pre-trained weight file (Only for elmo model)
https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

## Download NLTK punkt
```
python3 -m nltk.downloader -d $HOME/nltk_data punkt
```
or
```
python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
```

## Run `convert_msmarco.py`
```
python convert_msmarco.py {source file} {destination file}
python convert_msmarco.py ./train_v1.1.json train.tsv # for example
```
It should create `train.tsv`, `dev.tsv`, and `test.tsv, and only keep querys that query_type is "description". 
It does a fair amount of preprocessing therefore converting data to cntk text format reader starts from these files

## Run `tsv2ctf.py`
It creates a `vocabs.pkl` and `train.ctf`, `val.ctf`, and `dev.ctf`
The data is ready now. 

## use config_*.py to set training configuration
## Run 'train_pm.py' in GPU
This train code mainly refers to https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow
example
```
python train_pm.py -net BiDAF -logfile bidaf -model -bidaf --gpu 0
python train_pm.py -net RNet --gpu 1
```
## get result
```
python train_pm.py -model {saved model name} -test dev.tsv
```

MIT License