# 说明
该代码实现了一个基本的Seq2Seq模型，包括以下部分：

- Encoder
- State Vector
- Decoder

该repo下共有三个文件：

- ```Seq2seq_char.ipynb```是jupyter notebook可执行文件（推荐使用）
- ```Seq2seq_char.html```是html文件，方便查看代码结果
- ```Seq2seq_char.py```是由jupyter notebook转化的py文件

# 版本

- ```python 3```
- ```tensorflow 1.1```

# 代码中涉及到的function说明

- ```tf.contrib.layers.embed_sequence```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence>
	- 说明：对序列数据执行embedding操作，输入```[batch_size, sequence_length]```的tensor，返回```[batch_size, sequence_length, embed_dim]```的tensor。
	- 例子：
	
			features = [[1,2,3],[4,5,6]]
			outputs = tf.contrib.layers.embed_sequence(features, vocab_size, embed_dim)
			# 如果embed_dim=4，输出结果为
			[
				[[0.1,0.2,0.3,0.1],[0.2,0.5,0.7,0.2],[0.1,0.6,0.1,0.2]],
				[[0.6,0.2,0.8,0.2],[0.5,0.6,0.9,0.2],[0.3,0.9,0.2,0.2]]
			]
			
	  
- ```tf.strided_slice```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/strided_slice>
	- 说明：对传入的tensor执行切片操作，返回切片后的tensor。主要参数```input_, start, end, strides```，```strides```代表切片步长。
	- 例子：
	
			# 'input' is [[[1, 1, 1], [2, 2, 2]],
			#             [[3, 3, 3], [4, 4, 4]],
			#             [[5, 5, 5], [6, 6, 6]]]
			tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
			# 上面一行代码中[1,0,0]分别代表原数组三个维度的切片起始位置，[2,1,3]代表结束位置。
			[1,1,1]代表切片步长，表示在三个维度上切片步长都为1。我们的原始输入数据为3 x 2 x 3，
			通过参数我们可以得到，第一个维度上切片start=1,end=2，
			第二个维度start=0, end=1，第三个维度start=0, end=3。
			我们从里面的维度来看，原始数据的第三个维度有三个元素，切片操作start=0,end=3,stride=1，代表第三个维度上的元素我们全部保留。
			同理，在第二个维度上，start=0, end=1, stride=1，代表第二个维度上只保留第一个切片，这样我们就只剩下[[[1,1,1]],[[3,3,3]],[[5,5,5]]]。
			接着我们看第一个维度，start=1, end=2, stride=1代表只取第二个切片，因此得到[[[3,3,3]]。以下两个例子同理。
			
			tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
                                                               [4, 4, 4]]]
			tf.strided_slice(input, [1, -1, 0], [2, -3, 3], [1, -1, 1]) ==>[[[4, 4, 4],
                                                                 [3, 3, 3]]]

- ```tf.contrib.rnn.MultiRNNCell```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell>
	- 说明：对RNN单元按序列堆叠。接受参数为一个由RNN cell组成的list。
	- 例子：
			
			# rnn_size代表一个rnn单元中隐层节点数量，layer_nums代表堆叠的rnn cell个数
			lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
			composed_cell = tf.contrib.rnn.MultiRNNCell([lstm for _ in range(num_layers)])
			# 上面这种写法在tensorflow1.0中是可以运行的，但在tensorflow1.1版本中，以上构造的lstm单元不允许复用，要重新生成新的对象，因此在源码中，函数中嵌套了一个定义cell的函数，从而保证每次生成新的对象实例。
			def get_lstm(rnn_size):
				lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
				return lstm
			composed_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(num_layers)])

- ```tf.nn.dynamic_rnn```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>
	- 说明：构建RNN，接受动态输入序列。返回RNN的输出以及最终状态的tensor。```dynamic_rnn```与```rnn```的区别在于，```dynamic_rnn```对于不同的batch，可以接收不同的```sequence_length```，例如，第一个batch是```[batch_size,10]```，第二个batch是```[batch_size,20]```。而rnn只能接收定长的```sequence_length```。
	- 例子：
	
			output, state = tf.nn.dynamic_rnn(cell, inputs)

- ```tf.tile```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/tile>
	- 说明：对输入的tensor进行复制，返回复制后的tensor。主要参数是input和multiples。
	- 例子：
			
			# 伪代码
			input = [a, b, c, d]
			output = tf.tile(input, 2)
			# output = [a, b, c, d, a, b, c, d]
			
			input = [[1,2,3], [4,5,6]]
			output = tf.tile(input, [2, 3])
			# output = [[1,2,3,1,2,3,1,2,3],
						  [4,5,6,4,5,6,4,5,6],
						  [1,2,3,1,2,3,1,2,3],
						  [4,5,6,4,5,6,4,5,6]]
			
	
- ```tf.fill```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/fill>
	- 说明：主要参数为dims和value，构造一个由value填充的形状为dims的tensor。
	- 例子：

			tf.fill([2,3],9) => [[9,9,9],[9,9,9]]

- ```tf.contrib.seq2seq.TrainingHelper```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper>
	- 说明：Decoder端用来训练的函数。这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。主要参数是```inputs```和```sequence_length```。返回helper对象，可以作为BasicDecoder函数的参数。
	- 例子：

			training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

- ```tf.contrib.seq2seq.BasicDecoder```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder>
	- 说明：生成基本解码器对象
	- 例子：
			
			# cell为RNN层，training_helper是由TrainingHelper生成的对象，
			encoder_state是RNN的初始状态tensor，
			output_layer代表输出层，它是一个tf.layers.Layer的对象。
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
- ```tf.contrib.seq2seq.dynamic_decode```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode>
	- 说明：对decoder执行dynamic decoding。通过```maximum_iterations```参数定义最大序列长度。
			
			
- ```tf.contrib.seq2seq.GreedyEmbeddingHelper```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper>
	- 说明：它和```TrainingHelper```的区别在于它会把t-1下的输出进行embedding后再输入给RNN。

- ```tf.sequence_mask```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/sequence_mask>
	- 说明：对tensor进行mask，返回True和False组成的tensor
	- 例子：
	
			# 伪代码
			tf.sequence_mask([1,3,2],5) => 
			[[True, False, False, False, False],
			[True, True, True, False, False],
			[True, True, False, False, False]]
			# 其中dtype默认是tf.bool，在我们的代码中使用tf.float32，这是为后面计算loss生成权重。

- ```tf.contrib.seq2seq.sequence_loss```
	- 链接：<https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss>
	- 说明：对序列logits计算加权交叉熵。
	- 例子：
			
			# training_logits是输出层的结果，targets是目标值，masks是我们使用tf.sequence_mask计算的结果，在这里作为权重，也就是说我们在计算交叉熵时不会把<PAD>计算进去。
			cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)