#A PyTorch based demo for paraphrase identification.
## Introduction
Paraphrase identification is an important NLP task, which can be used to improve many other NLP tasks such as information retrieval and question answering. It is a kind of text classification, which is to judge whether two sentences have the same meaning. This demo is designed to finish paraphrase identification task on Microsoft Research Paraphrase Corpus(MSRP) dataset, which the sentences are extracted from news.


## Dataset
The [MSRP](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset contains 5800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship.


## Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- torchtext==0.2.3
- torch==0.4.1


## Runtime requirements
Using the default parameters: 
- GPU mode need about 600 MB cuda memory.
- CPU mode takes tiny CPU utilization and less than 300 MB of main memory usage. 


## Usage

```
python2/python3 paraphrase_identification.py --lr 0.001 --batch_size 64 --max_iter 5000 --data_dir data --model_dir model --model_name best_model.pth
```

The code will download the dataset automatically.


## Results
With the default hypeparameters, you would get an accuracy of 67.77%.


## FAQ
#### I am getting out-of-memory errors, what is wrong?
You are likely to encounter out-of-momery issues using the default parameters if your GPU momery less than 2GB. 
The factors that affect memory usage are batch_size, hidden_size and so on, you can try to decrease these hypeparameters.


## Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [torchtext team](https://github.com/pytorch/text)  for Python library<br>


## Reference
* [Unsupervised construction of large paraphrase corpora: exploiting massively parallel news sources.](https://www.microsoft.com/en-us/research/publication/unsupervised-construction-of-large-paraphrase-corpora-exploiting-massively-parallel-news-sources/)

## License
MIT
