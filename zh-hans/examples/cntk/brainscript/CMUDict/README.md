[English](/examples/cntk/brainscript/CMUDict/README.md)

此样例演示了通过CNTK来使用CMUDict字典，通过序列到序列的注意力模型来实现书写位到音素（字母到声音）的转换。

该代码支持多种替代配置。 当前的配置如下：

* 3隐藏层的单向LSTM编码网络，隐藏层维度均为512。
* 3隐藏层的单向LSTM解码网络，隐藏层维度均为512。
* 编码器状态通过注意力的方法传递给解码器，投影维度为128，最大输入长度是20个标记。
* 嵌入已被禁用（因为任务、字母和音素的'词汇量'非常小）
* 集束（beam）解码器的宽度为3

## 如何使用

根据需要，修改G2P.cntk文件的内容：

* pathnames
* deviceId 用来指定到某个CPU (-1) 或 GPU (>=0或"auto")

运行：

* 命令行： ```cntk  configFile=Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=g2p```
* VS调试： ```configFile=$(SolutionDir)Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=$(SolutionDir)Examples/SequenceToSequence/CMUDict```