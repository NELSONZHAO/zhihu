# 说明
由于部分同学反映原来的代码运行有问题。因此我重新添加了适用于TensorFlow1.0版本的代码，对应的文件为：

- anna_lstm-tf1.0.ipynb
- anna_lstm-tf1.0.html

该版本的代码与之前版本的变化如下：

- 修改```tf.nn.rnn_cell```为```tf.contrib.rnn```
- 修改```tf.concat(1, lstm_output)```为```tf.concat(lstm_output, axis=1)```

如需运行1.0版本代码，请在执行floyd时，使用：
```floyd run --gpu --env tensorflow-1.0 --mode jupyter```