"""
This file is used to load pre-trained VGG model
"""
# coding: utf-8

import numpy as np
import scipy.io
import tensorflow as tf
import utils

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
VGG_FILENAME = "imagenet-vgg-verydeep-19.mat"
EXPECTED_BYTES = 534904783  # 文件大小


class VGG(object):
    def __init__(self, input_img):
        # 下载文件
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        # 加载文件
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)["layers"]
        self.input_img = input_img
        # VGG在处理图像时候会将图片进行mean-center，所以我们首先要计算RGB三个channel上的mean
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def _weights(self, layer_idx, expected_layer_name):
        """
        获取指定layer层的pre-trained权重
        
        :param layer_idx: VGG中的layer id
        :param expected_layer_name: 当前layer命名
        :return: pre-trained权重W和b
        """
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        # 当前层的名称
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name, print("Layer name error!")

        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """
        采用relu作为激活函数的卷积层
        
        :param prev_layer: 前一层网络
        :param layer_idx: VGG中的layer id
        :param layer_name: 当前layer命名
        """
        with tf.variable_scope(layer_name):
            # 获取当前权重（numpy格式）
            W, b = self._weights(layer_idx, layer_name)
            # 将权重转化为tensor（由于我们不需要重新训练VGG的权重，因此初始化为常数）
            W = tf.constant(W, name="weights")
            b = tf.constant(b, name="bias")
            # 卷积操作
            conv2d = tf.nn.conv2d(input=prev_layer,
                                  filter=W,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")
            # 激活
            out = tf.nn.relu(conv2d + b)
        setattr(self, layer_name, out)

    def avgpool(self, prev_layer, layer_name):
        """
        average pooling层（这里参考了原论文中提到了avg-pooling比max-pooling效果好，所以采用avg-pooling）
        
        :param prev_layer: 前一层网络（卷积层）
        :param layer_name: 当前layer命名
        """
        with tf.variable_scope(layer_name):
            # average pooling
            out = tf.nn.avg_pool(value=prev_layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

        setattr(self, layer_name, out)

    def load(self):
        """
        加载pre-trained的数据
        """
        self.conv2d_relu(self.input_img, 0, "conv1_1")
        self.conv2d_relu(self.conv1_1, 2, "conv1_2")
        self.avgpool(self.conv1_2, "avgpool1")
        self.conv2d_relu(self.avgpool1, 5, "conv2_1")
        self.conv2d_relu(self.conv2_1, 7, "conv2_2")
        self.avgpool(self.conv2_2, "avgpool2")
        self.conv2d_relu(self.avgpool2, 10, "conv3_1")
        self.conv2d_relu(self.conv3_1, 12, "conv3_2")
        self.conv2d_relu(self.conv3_2, 14, "conv3_3")
        self.conv2d_relu(self.conv3_3, 16, "conv3_4")
        self.avgpool(self.conv3_4, "avgpool3")
        self.conv2d_relu(self.avgpool3, 19, "conv4_1")
        self.conv2d_relu(self.conv4_1, 21, "conv4_2")
        self.conv2d_relu(self.conv4_2, 23, "conv4_3")
        self.conv2d_relu(self.conv4_3, 25, "conv4_4")
        self.avgpool(self.conv4_4, "avgpool4")
        self.conv2d_relu(self.avgpool4, 28, "conv5_1")
        self.conv2d_relu(self.conv5_1, 30, "conv5_2")
        self.conv2d_relu(self.conv5_2, 32, "conv5_3")
        self.conv2d_relu(self.conv5_3, 34, "conv5_4")
        self.avgpool(self.conv5_4, "avgpool5")