# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,     
    Layer, 
    Dense, 
    Activation,
    DepthwiseConv2D,
    BatchNormalization,
    GlobalAveragePooling2D
    )

# hard swish
def sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def swish(x):
    return x * sigmoid(x)

# wrap hard swish as a Layer 
class Swish(Layer):
    """
    docstring
    """
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, x):
        return swish(x)

class SEBlock(Layer):
    def __init__(self, input_channels, r=4):  # r=16
        super(SEBlock, self).__init__()
        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(units=input_channels // r)
        self.fc2 = Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = inputs * branch

        return output


class BottleNeck(Layer):
    def __init__(self, in_size, exp_size, out_size, stride, is_se_existing, NL, ksize, name=None):
        super(BottleNeck, self).__init__(name=name)
        self.stride = stride
        self.in_size = in_size
        self.out_size = out_size
        self.is_se_existing = is_se_existing
        self.NL = NL
        self.conv1 = Conv2D(filters=exp_size, 
                            kernel_size=(1, 1),
                            strides=1,
                            padding="same")
        self.bn1 = BatchNormalization()
        self.dwconv = DepthwiseConv2D(kernel_size=(ksize, ksize),
                                      strides=stride,
                                      padding="same")
        self.bn2 = BatchNormalization()
        self.se = SEBlock(input_channels=exp_size)
        self.conv2 = Conv2D(filters=out_size,
                            kernel_size=(1, 1), 
                            strides=1,
                            padding="same")
        self.bn3 = BatchNormalization()
        self.linear = Activation(tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.NL == "HS":
            x = swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)
        
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        if self.NL == "HS":
            x = swish(x)
        elif self.NL == "RE":
            x = tf.nn.relu6(x)

        if self.is_se_existing:
            x = self.se(x)

        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = tf.keras.layers.add([x, inputs])

        return x


if __name__ == "__main__":
    print("haha")