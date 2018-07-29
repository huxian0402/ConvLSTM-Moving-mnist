# ConvLSTM-Moving-mnist
Using convlstm to prediction moving mnist dataset.

使用其中一个序列，迭代训练，收敛。 前10帧--第11帧，...，第10-19帧预测第20帧。
输入mnist序列            三层conv        一层BasicConvLstmCel  三层conv flatten
（1，10，64，64，1） （1，10，64，64，16）（1，1，64，64，1024） （1，1，64，64，1）

训练过程loss变化:

![image](https://github.com/huxian0402/ConvLSTM-Moving-mnist/blob/master/loss.png)
