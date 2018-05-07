# Tensorflow 中的端到端内存网络

[English](/examples/tensorflow/MemN2N/README.md)

Tensorflow的[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4)语言模型（见第5章）的实现。 Facebook的原始torch代码在[这里](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model)。

![替换标签](http://i.imgur.com/nv89JLc.png)

## 先决条件

需安装[Tensorflow](https://www.tensorflow.org/)。 在`data`文件夹中有Penn Tree Bank (PTB) 语料的样例集，这是一个常用的衡量多种模型质量的基准数据集。 也可将自己的数据按照[这样](data/)格式化后使用此模型。

当使用docker图像tensorflw/tensorflow:latest-gpu时，还需要Python future 包。

    $ pip install future
    

如果要使用`--show True`选项，还需要安装Python `progress`包。

    $ pip install progress
    

## 用法

要训练6个跃点，且内存大小为100的模型，运行下列命令：

    $ python main.py --nhop 6 --mem_size 100
    

查看所有训练选项，运行：

    $ python main.py --help
    

输出为（下为译文）：

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
    

（可选）如果要查看进度条，用`pip`安装`progress`：

    $ pip install progress
    $ python main.py --nhop 6 --mem_size 100 --show True
    

训练完成后用下面的命令来测试并验证：

    $ python main.py --is_test True --show True
    

训练输出样例：

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

Penn Treebank语料测试集上的困惑度。

| 隐藏数量 | 跃点数量 | 存储大小 | MemN2N (Sukhbaatar 2015) |    本代码库     |
|:----:|:----:|:----:|:------------------------:|:-----------:|
| 150  |  3   | 100  |           122            |     129     |
| 150  |  6   | 150  |           114            | in progress |

## 作者

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)

## 许可证

MIT许可证