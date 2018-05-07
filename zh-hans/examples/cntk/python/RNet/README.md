# bidaf入门说明

## 下载微软MARCO数据集

    wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
    wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
    wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
    

## 下载GloVe向量

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    

## 下载NLTK punkt

    python3 -m nltk.downloader -d $HOME/nltk_data punkt
    

or

    python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
    

## 运行`convert_msmarco.py`

It should create `train.tsv`, `dev.tsv`, and `test.tsv`, and only keep querys that query_type is "description".

It does a fair amount of preprocessing therefore converting data to cntk text format reader starts from these files

## 运行`tsv2ctf.py`

It creates a `vocabs.pkl` and `train.ctf`, `val.ctf`, and `dev.ctf`

The data is ready now.

## 在GPU上运行'rnetmodel.py'

you can see the model structure from here

## 在GPU上运行'train_rnet.py'

this train code mainly from here https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow