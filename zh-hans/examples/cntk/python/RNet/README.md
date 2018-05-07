# bidaf入门说明

[English](/examples/cntk/python/RNet/README.md)

## 下载微软MARCO数据集

    wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
    wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
    wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
    

## 下载GloVe向量

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    

## 下载NLTK punkt

    python3 -m nltk.downloader -d $HOME/nltk_data punkt
    

或者

    python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
    

## 运行`convert_msmarco.py`

运行后会创建`train.tsv`，`dev.tsv`以及`test.tsv`，并会在query_type为"description"时保留查询。

这会进行大量的数据预处理来将数据转换为cntk的文本格式。

## 运行`tsv2ctf.py`

运行后会创建`vocabs.pkl`和`train.ctf`，`val.ctf`，以及`dev.ctf`。

到这里，数据就准备完成了。

## 在GPU上运行'rnetmodel.py'

在这里面可以看到模型的结构

## 在GPU上运行'train_rnet.py'

训练代码主要来源于下面的位置 https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow