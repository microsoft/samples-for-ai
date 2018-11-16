#A PyTorch based demo for text classification.

### Requirements
- numpy==1.15.3
- torchtext==0.2.3
- torch==0.4.1


## Usage

```
python2/python3 slot_tagging.py --lr 0.001 --batch_size 64 --max_iter 5000 --data_dir data --model_dir model --model_name best_model.pth
```

This downloads the following data automatically:
  - [Airline Travel Information System DataSet](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) (This data set consists of 20000 messages taken from 20 newsgroups)

### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [torchtext team](https://github.com/pytorch/text) for Python library<br>
* [JointSLU] (https://github.com/yvchen/JointSLU) for the ATIS datasets

### License
MIT
