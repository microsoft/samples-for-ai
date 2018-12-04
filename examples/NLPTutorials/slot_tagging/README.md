#A PyTorch based demo for slot tagging.
## Introduction
Slot tagging is an important NLP task, which can be used to improve many other NLP tasks such as named entity recognition and question answering. This demo is designed to finish slot tagging task on Airline Travel Information Service (ATIS) dataset. For example, the cities and the time would be noted.


## Dataset
The [Airline Travel Information Service DataSet] (http://aclweb.org/anthology/H90-1020) is a commonly used dataset for slot filling, which is about he air travel domain.


## Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- torchtext==0.2.3
- torch==0.4.1


## Runtime requirements
Using the default parameters: 
- GPU mode need about 600 MB cuda memory.
- CPU mode takes tiny CPU utilization and less than 400 MB of main memory usage. 


## Usage

```
python2/python3 slot_tagging.py --lr 0.001 --batch_size 64 --max_iter 5000 --data_dir data --model_dir model --model_name best_model.pth
```

The code will download the dataset automatically.


## Results
With the default hypeparameters, you would get a label accuracy of 98.11%.


## FAQ
#### I am getting out-of-memory errors, what is wrong?
You are likely to encounter out-of-momery issues using the default parameters if your GPU momery less than 2GB. 
The factors that affect memory usage are batch_size, hidden_size and so on, you can try to decrease these hypeparameters.


## Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [torchtext team](https://github.com/pytorch/text) for Python library<br>
* [JointSLU] (https://github.com/yvchen/JointSLU) for the ATIS datasets


## License
MIT
