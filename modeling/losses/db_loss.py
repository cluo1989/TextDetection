'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-12-02 16:25:52
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-16 17:27:17
FilePath: /dbnet_plus/modeling/losses/db_loss.py
Description: 
'''
from tensorflow.keras.layers import Layer
from modeling.losses.basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(Layer):
    def __init__(self, 
        params,
        eps=1e-6,
        **kwargs
        ):
        super(DBLoss, self).__init__()
        balance_loss = params['balance_loss']
        main_loss_type = params['main_loss_type']
        ohem_ratio = params['ohem_ratio']
        
        self.alpha = params['alpha']
        self.beta = params['beta']
        
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio
        )

    def call(self, predicts, labels):
        predict_maps = predicts['maps']
        # label_thresh_map, label_thresh_mask, \
        #     label_shrink_map, label_shrink_mask = labels[1:]
        #OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed in Graph execution
        label_thresh_map = labels[...,1], 
        label_thresh_mask = labels[...,2]
        label_shrink_map = labels[...,3]
        label_shrink_mask = labels[...,4]
        
        shrink_maps = predict_maps[...,0]
        thresh_maps = predict_maps[...,1]
        binary_maps = predict_maps[...,2]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map, label_shrink_mask)
        loss_thresh_maps = self.l1_loss(thresh_maps, label_thresh_map, label_thresh_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)

        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_thresh_maps = self.beta * loss_thresh_maps
        loss_all = loss_shrink_maps + loss_thresh_maps + loss_binary_maps

        losses = {
            'loss': loss_all, 
            'loss_shrink_maps': loss_shrink_maps,
            'loss_thresh_maps': loss_thresh_maps,
            'loss_binary_maps': loss_binary_maps,
        }

        return losses
