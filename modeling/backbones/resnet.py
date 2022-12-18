'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-29 17:28:10
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-15 15:20:13
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
        self.is_3x3 = True  # False
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

        filters = [32, 64, 128, 256]  #[64, 128, 256, 512]

        # first conv Block
        if is_3x3 == False:
            conv = Conv2D(64, (7, 7), strides=(1, 1), padding="same", name="conv1")(inputs)  # 32->64
            conv = BatchNormalization(name="conv1_bn")(conv)
            conv = Activation("relu")(conv)

        else:
            conv = Conv2D(32, (3, 3), strides=(2, 2), padding="same", name="conv1_1")(inputs) # 16->32
            conv = BatchNormalization(name="conv1_1_bn")(conv)
            conv = Activation("relu")(conv)

            conv = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name="conv1_2")(conv)
            conv = BatchNormalization(name="conv1_2_bn")(conv)
            conv = Activation("relu")(conv)

            conv = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv1_3")(conv)
            conv = BatchNormalization(name="conv1_3_bn")(conv)
            conv = Activation("relu")(conv)


        conv = MaxPool2D((3, 3), strides=(2, 2), padding="same")(conv)
        inplane = 64  # 32->64

        # 4 stages
        feat_maps = []        
        for block in range(len(depth)):
            for i in range(depth[block]):
                if i == 0 and block != 0:
                    stride = 2
                else:
                    stride = 1

                if layers >= 50:
                    conv = BottleNeck(inplane, filters[block], stride=stride)(conv)
                    inplane = filters[block] * BottleNeck.expansion
                else:
                    conv = BasicBlock(inplane, filters[block], stride=stride)(conv)
                    inplane = filters[block] * BasicBlock.expansion

            feat_maps.append(conv)

        return feat_maps


if __name__ == "__main__":
    # check shape
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    inputs = Input(shape=[320, 320, 3], dtype="float32", name="images")
    outputs = ResNet({"num_layers":50})(inputs) 
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    print('\n'.join([str(s) for s in model.output_shape[0]]))
