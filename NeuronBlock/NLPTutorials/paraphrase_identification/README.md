#A PyTorch based demo for text classification.

### Requirements
- numpy==1.15.3
- torchtext==0.2.3
- torch==0.4.1


## Usage

```
python2/python3 paraphrase_identification.py --lr 0.001 --batch_size 64 --max_iter 5000 --data_dir data --model_dir model --model_name best_model.pth
```

This downloads the following data automatically:
  - [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (This data set consists of 5800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship.)

### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [torchtext team](https://github.com/pytorch/text)  for Python library<br>

### Reference
* [Unsupervised construction of large paraphrase corpora: exploiting massively parallel news sources.](https://www.microsoft.com/en-us/research/publication/unsupervised-construction-of-large-paraphrase-corpora-exploiting-massively-parallel-news-sources/)

### License
MIT
