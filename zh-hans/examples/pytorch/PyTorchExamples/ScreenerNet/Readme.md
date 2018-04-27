这是 [ScreenerNet: Learning Self-Paced Curriculum for Deep Neural Networks](https://arxiv.org/abs/1801.00904) 的实现。

# 必要内容

如果要在pascal voc2012数据集上训练，确保下载了它，并放在了PWD/VOC2012目录下。

# 如何运行

输入 `python snet.py -h`  
查看所有训练和测试的选项。

例如 `python snet.py train mnist
python snet.py test mnist --modelname=your_model_name`