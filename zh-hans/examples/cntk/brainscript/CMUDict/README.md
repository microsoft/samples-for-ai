This example demonstrates the use of CNTK for grapheme-to-phoneme (letter-to-sound) conversion using a sequence-to-sequence model with attention, using the CMUDict dictionary.

The code supports a number of alternative configurations. As configured currently, it implements

* 3隐藏层的单向LSTM编码网络，隐藏层维度均为512。
* 3隐藏层的单向LSTM解码网络，隐藏层维度均为512。
* 编码器状态通过注意力的方法传递给解码器，投影维度为128，最大输入长度是20个标记。
* 嵌入已被禁用（因为任务、字母和音素的'词汇量'非常小）
* 集束（beam）解码器的宽度为3

## 如何使用

Modify the following in G2P.cntk as needed:

* pathnames
* deviceId 用来指定到某个CPU (-1) 或 GPU (>=0或"auto")

Run:

* 命令行： ```cntk  configFile=Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=g2p```
* VS调试： ```configFile=$(SolutionDir)Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=$(SolutionDir)Examples/SequenceToSequence/CMUDict```