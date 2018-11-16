#A PyTorch based demo for language model.

### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- torchtext==0.2.3


## Usage

```
python/python3 language_model.py --batch_size 20 --epochs 20 --gen_word_len 15
```

This downloads the following data automatically:
  - [WikiText2 Data Set]('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')

### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>

### License
MIT
