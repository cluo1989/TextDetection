'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-30 10:50:54
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-18 02:20:09
FilePath: /dbnet_plus/modeling/architectures/dbnet.py
Description: Architecture of DBNet++
'''
# coding: utf-8
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from utils.util import create_module


class DBNet_Plus(object):
    """DBNet++ Model class.

    build model architecture according to params.
    """
    def __init__(self, params):
        '''parse params
        '''
        super(DBNet_Plus, self).__init__()
        global_params = params['Global']
        self.input_shape = global_params['INPUT_SHAPE']

        backbone_params = params['Backbone']
        neck_params = params['Neck']
        head_params = params['Head']        
        loss_params = params['Loss']

        self.backbone = create_module(backbone_params['function'])(backbone_params)
        self.neck = create_module(neck_params['function'])(neck_params)
        self.head = create_module(head_params['function'])(head_params)
        self.loss = create_module(loss_params['function'])(loss_params)
        
    def __call__(self, mode):
        '''build model
        '''
        assert mode in ['predict', 'train'], \
            "invalid mode, select one of ['predict', 'train']"

        images = Input(shape=self.input_shape, dtype="float32", name="images")
        conv_feats = self.backbone(images)
        neck_feats = self.neck(conv_feats)
        predicts = self.head(neck_feats)

        if mode == "train":
            labels = Input(shape=[None, None, 5], dtype="float32", name="labels")
            loss = self.loss(predicts, labels)
            model = Model(inputs=[images, labels], outputs=[loss])
            return model
        else:
            model = Model(inputs=images, outputs=predicts)
            return model
            