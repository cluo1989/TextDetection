'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-29 17:28:10
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-02 14:39:17
FilePath: /dbnet_plus/modeling/backbones/resnet.py
Description: Implementation of ResNet
'''
# coding: utf-8
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization
from modeling.backbones.blocks.residual_block import BasicBlock, BottleNeck


class ResNet(object):
    """ ResNet Model class

    implementation of ResNetV1 according to paper:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.num_layers = params['num_layers']
        self.is_3x3 = False
        supported_layers = [18, 34, 50, 101, 152]
        assert self.num_layers in supported_layers, \
            "unsupported layers"

    def __call__(self, inputs):
        'forward pass'
        is_3x3 = self.is_3x3
        layers = self.num_layers

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]

        filters = [32, 64, 128, 256]

        # first conv Block
        if is_3x3 == False:
            conv = Conv2D(32, (7, 7), strides=(1, 1), padding="same", name="conv1")(inputs)
            conv = BatchNormalization(name="conv1_bn")(conv)
            conv = Activation("relu")(conv)

        else:
            conv = Conv2D(16, (3, 3), strides=(1, 1), padding="same", name="conv1_1")(inputs)
            conv = BatchNormalization(name="conv1_1_bn")(conv)
            conv = Activation("relu")(conv)

            conv = Conv2D(16, (3, 3), strides=(1, 1), padding="same", name="conv1_2")(conv)
            conv = BatchNormalization(name="conv1_2_bn")(conv)
            conv = Activation("relu")(conv)

            conv = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name="conv1_3")(conv)
            conv = BatchNormalization(name="conv1_3_bn")(conv)
            conv = Activation("relu")(conv)


        conv = MaxPool2D((3, 3), strides=(2, 2))(conv)
        inplane = 32

        # 4 stages
        feat_maps = []
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    conv = BottleNeck(inplane, filters[block], stride=stride)(conv)
                    feat_maps.append(conv)
                    inplane = filters[block] * BottleNeck.expansion
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    conv = BasicBlock(inplane, filters[block], stride=stride)(conv)
                    feat_maps.append(conv)
                    inplane = filters[block] * BasicBlock.expansion

        return feat_maps
        