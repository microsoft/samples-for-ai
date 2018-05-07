# Tensorflow 中的端到端内存网络

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modeling (see Section 5). The original torch code from Facebook can be found [here](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model).

![alt tag](http://i.imgur.com/nv89JLc.png)

## 先决条件

This code requires [Tensorflow](https://www.tensorflow.org/). There is a set of sample Penn Tree Bank (PTB) corpus in `data` directory, which is a popular benchmark for measuring quality of these models. But you can use your own text data set which should be formated like [this](data/).

When you use docker image tensorflw/tensorflow:latest-gpu, you need to python package future.

    $ pip install future
    

If you want to use `--show True` option, you need to install python package `progress`.

    $ pip install progress
    

## 用法

To train a model with 6 hops and memory size of 100, run the following command:

    $ python main.py --nhop 6 --mem_size 100
    

To see all training options, run:

    $ python main.py --help
    

which will print:

    用法: main.py [-h] [--edim EDIM] [--lindim LINDIM] [--nhop NHOP]
                  [--mem_size MEM_SIZE] [--batch_size BATCH_SIZE]
                  [--nepoch NEPOCH] [--init_lr INIT_LR] [--init_hid INIT_HID]
                  [--init_std INIT_STD] [--max_grad_norm MAX_GRAD_NORM]
                  [--data_dir DATA_DIR] [--data_name DATA_NAME] [--show SHOW]
                  [--noshow]
    
    可选参数:
      -h, --help            显示帮助信息并退出
      --edim EDIM           内部状态维度 [150]
      --lindim LINDIM       部分线性状态 [75]
      --nhop NHOP           跃点数量 [6]
      --mem_size MEM_SIZE   内存大小 [100]
      --batch_size BATCH_SIZE
                            训练中的批处理大小 [128]
      --nepoch NEPOCH       训练中的批次数量 [100]
      --init_lr INIT_LR     初始学习率 [0.01]
      --init_hid INIT_HID   初始内部状态值 [0.1]
      --init_std INIT_STD   初始权重标准差 [0.05]
      --max_grad_norm MAX_GRAD_NORM
                            最大梯度正则值 [50]
      --checkpoint_dir CHECKPOINT_DIR
                            检查点文件夹 [checkpoints]
      --data_dir DATA_DIR   数据文件夹 [data]
      --data_name DATA_NAME
                            数据集名称 [ptb]
      --is_test IS_TEST     测试为True，训练为False [False]
      --nois_test
      --show SHOW           打印进度 [False]
      --noshow
    

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --nhop 6 --mem_size 100 --show True
    

After training is finished, you can test and validate with:

    $ python main.py --is_test True --show True
    

The training output looks like:

    $ python main.py --nhop 6 --mem_size 100 --show True
    Read 929589 words from data/ptb.train.txt
    Read 73760 words from data/ptb.valid.txt
    Read 82430 words from data/ptb.test.txt
    {'batch_size': 128,
    'data_dir': 'data',
    'data_name': 'ptb',
    'edim': 150,
    'init_hid': 0.1,
    'init_lr': 0.01,
    'init_std': 0.05,
    'lindim': 75,
    'max_grad_norm': 50,
    'mem_size': 100,
    'nepoch': 100,
    'nhop': 6,
    'nwords': 10000,
    'show': True}
    I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 12
    I tensorflow/core/common_runtime/direct_session.cc:45] Direct session inter op parallelism threads: 12
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 507.3536108810464, 'epoch': 0, 'valid_perplexity': 285.19489755719286, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 218.49577035468886, 'epoch': 1, 'valid_perplexity': 231.73457031084268, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 163.5527845871247, 'epoch': 2, 'valid_perplexity': 175.38771414841014, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 136.1443535538306, 'epoch': 3, 'valid_perplexity': 161.62522958776597, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 119.15373237680929, 'epoch': 4, 'valid_perplexity': 149.00768378137946, 'learning_rate': 0.01}
    Training |##############                  | 44.0% | ETA: 378s
    

## 性能

The perplexity on the test sets of Penn Treebank corpora.

| 隐藏数量 | 跃点数量 | 存储大小 | MemN2N (Sukhbaatar 2015) |    本代码库     |
|:----:|:----:|:----:|:------------------------:|:-----------:|
| 150  |  3   | 100  |           122            |     129     |
| 150  |  6   | 150  |           114            | in progress |

## 作者

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)

## 许可证

MIT License