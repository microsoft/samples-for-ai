# A PyTorch based demo for query-passage similarity model.

### Introduction
The query-passage similarity model is to compute the similarity between querys and passages, which is a crucial subtask of question answering (QA). 

The goal of this demo is to learn to determine whether the passage(sentence) is a correct answer to the corresponding query.

### Dataset
[WikiQA corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence(passage) pairs.

The dataset includes 3,047 questions and 29,258 sentences, where 1,473 sentences were labeled as answer sentences to their corresponding questions. 

More detail of this corpus can be found in EMNLP-2015 paper: [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://aclweb.org/anthology/D15-1237) [Yang et al. 2015]. 

### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- torchtext==0.2.3
- scikit-learn==0.20.0

### Runtime requirements
Using the default parameters: 
- GPU mode need about 1GB cuda memory.
- CPU mode takes tiny CPU utilization and about 500 MB of main memory usage. 

### Usage

```
python/python3 qp_similarity.py --batch_size 64 --epochs 15
```

This downloads the following data automatically:
  - [WikiQACorpus](https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip)

### Result
With the method and default parameters we provide, you will get the following results on test dataset:
```
Test Loss: 0.5476       AUC: 0.7462
```

### FAQ

#### I am getting out-of-memory errors, what is wrong?
You are likely to encounter out-of-momery issues using the default parameters if your GPU momery less than 1GB. 
The factors that affect memory usage are batch_size, hidden_dim, embed_dim, filters, and so on, you can try to reduce
these parameters.

### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>

### License
MIT
