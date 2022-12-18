'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-12-02 16:13:58
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-15 15:24:08
FilePath: /dbnet_plus/modeling/necks/fpn.py
Description: 
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Layer, Conv2D, Conv2DTranspose, UpSampling2D, Activation
    )


class ASFBlock(Layer):
    """Adaptive Scale Fusion 
    """
    def __init__(self, inter_channels, out_features_num=4):
        super(ASFBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = Conv2D(inter_channels, 3, padding='same')

        self.spatial_scale = keras.Sequential([
            Conv2D(1, 3, padding='same'), 
            Activation('relu'), 
            Conv2D(1, 1),
            Activation('sigmoid')
        ])

        self.channel_scale = keras.Sequential([
            Conv2D(self.out_features_num, 1),
            Activation('sigmoid')
        ])


    def call(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)                          # Nx80x80x256 -> Nx80x80x64        
        spatial_x = tf.reduce_mean(fuse_features, axis=-1, keepdims=True) # Nx80x80x1        
        attention_scores = self.spatial_scale(spatial_x) + fuse_features  # Nx80x80x1 + Nx80x80x64        
        attention_scores = self.channel_scale(attention_scores)           # Nx80x80x64 -> Nx80x80x4
        
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[..., i:i+1] * features_list[i])  # Nx80x80x64
        return tf.concat(out_list, axis=-1)  # Nx80x80x256
        

class DBFPN(Layer):
    """
    """
    def __init__(self, params, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = params['out_channels']
        self.use_asf = params['use_asf']
        print(f'---- out:{self.out_channels}, asf:{self.use_asf} ----')
        #weight_attr = KaimingUniform(), initializer

        # define layers, same output channel
        # default: kernel_initializer='glorot_uniform', bias_initializer='zeros'
        # default: strides=1, padding=0 ('valid')
        self.in2_conv = Conv2D(self.out_channels, 1)
        self.in3_conv = Conv2D(self.out_channels, 1)
        self.in4_conv = Conv2D(self.out_channels, 1)
        self.in5_conv = Conv2D(self.out_channels, 1)

        self.p5_conv = Conv2D(self.out_channels // 4, 3, padding='same')
        self.p4_conv = Conv2D(self.out_channels // 4, 3, padding='same')
        self.p3_conv = Conv2D(self.out_channels // 4, 3, padding='same')
        self.p2_conv = Conv2D(self.out_channels // 4, 3, padding='same')
        
        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels // 4)

            
    def upsample(self, x, scale_factor, mode='nearest'):
        # x = Conv2DTranspose(scale_factor, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D(scale_factor, interpolation=mode)(x)
        return x

    def call(self, feat_maps):
        # get multi scale feature maps
        c2, c3, c4, c5 = feat_maps

        
        in5 = self.in5_conv(c5) # 10x10
        in4 = self.in4_conv(c4) # 20x20
        in3 = self.in3_conv(c3) # 40x40
        in2 = self.in2_conv(c2) # 80x80

        out4 = in4 + self.upsample(in5, 2)  # 20x20
        out3 = in3 + self.upsample(out4, 2) # 40x40
        out2 = in2 + self.upsample(out3, 2) # 80x80

        p5 = self.p5_conv(in5)  # 10x10
        p4 = self.p4_conv(out4) # 20x20
        p3 = self.p3_conv(out3) # 40x40
        p2 = self.p2_conv(out2) # 80x80

        p5 = self.upsample(p5, 8) # 80x80 
        p4 = self.upsample(p4, 4) # 80x80
        p3 = self.upsample(p3, 2) # 80x80

        fuse = tf.concat([p5, p4, p3, p2], axis=-1) # pi: 80 x 80 x out_channels//4

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])
        
        return fuse