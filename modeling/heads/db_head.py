'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-12-02 16:20:39
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-15 15:44:20
FilePath: /dbnet_plus/modeling/heads/db_head.py
Description: 
'''
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, Conv2DTranspose, BatchNormalization, Activation
    )


class Head(Layer):
    """
    """
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()
        self.conv1 = Conv2D(in_channels//4, kernel_size=kernel_list[0], padding='same')
        self.conv_bn1 = BatchNormalization(name='conv_bn1')
        self.act1 = Activation('relu')

        self.conv2 = Conv2DTranspose(in_channels//4, kernel_size=kernel_list[1], strides=2)
        self.conv_bn2 = BatchNormalization(name='conv_bn2')
        self.act2 = Activation('relu')
        self.conv3 = Conv2DTranspose(1, kernel_size=kernel_list[2], strides=2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = tf.sigmoid(x)
        return x
        

class DBHead(Layer):
    def __init__(self, params, **kwargs):
        super(DBHead, self).__init__()
        self.in_channels = params['in_channels']
        self.k = params['k']

        self.binarize = Head(self.in_channels)
        self.thresh = Head(self.in_channels)

    def step_func(self, x, y):
        return tf.math.reciprocal(1 + tf.math.exp(- self.k * (x - y)))
        
    def call(self, x, training=True):
        shrink_maps = self.binarize(x)   # Nx80x80x256 -> Nx320x320x1
        if not training:
            return {'maps': shrink_maps}

        thresh_maps = self.thresh(x)     # Nx80x80x256 -> Nx320x320x1 
        binary_maps = self.step_func(shrink_maps, thresh_maps)  # Nx320x320x1
        print(tf.shape(thresh_maps), tf.shape(binary_maps), '-----------------')
        y = tf.concat([shrink_maps, thresh_maps, binary_maps], axis=-1)
        return {'maps': y}


# Conv2DTranspose output shape: [Hin,Win]->[Hout,Wout]
# same
# H′out=(Hin+stride[0]−1)/stride[0]
# W′out=(Win+stride[1]−1)/stride[1]

# valid 
# H′out=(Hin−1)∗strides[0]+dilations[0]∗(kernel_size[0]−1)+1
# W′out=(Win−1)∗strides[1]+dilations[1]∗(kernel_size[1]−1)+1