#A PyTorch based demo for text classification.

### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- Pillow==5.3.0
- requests==2.20.0
- scikit-learn==0.20.0
- scipy==1.1.0
- six==1.11.0
- sklearn==0.0
- torchtext==0.2.3
- tqdm==4.28.1
- urllib3==1.24


## Usage

```
python/python3 text_classification.py --model_name <TextCNN|LSTMSelfAttentionHighway> --batch_size 64 --epochs 10
```
#### Run on CPU

```
python/python3 text_classification.py --model_name <TextCNN|LSTMSelfAttentionHighway> --batch_size 64 --epochs 10 -device -1
```

This downloads the following data automatically:
  - [Twenty Newsgroups Data Set](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) (This data set consists of 20000 messages taken from 20 newsgroups)

### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [A pytorch implementation of CNNText classification](https://github.com/Shawn1993/cnn-text-classification-pytorch)
* [A pytorch implementation of LSTM + Self Attention classification](https://github.com/nn116003/self-attention-classification)

### Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* [A structured self-attention sentence embedding](https://arxiv.org/pdf/1703.03130.pdf)
* [Highway Network](https://arxiv.org/abs/1505.00387)

### License
MIT
