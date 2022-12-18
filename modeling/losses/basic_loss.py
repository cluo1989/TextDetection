'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-12-11 17:24:54
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-16 18:14:16
FilePath: /dbnet_plus/modeling/losses/basic_loss.py
Description: Loss
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError


class BCELoss(Layer):
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        # NONE, [batchsize, dim]
        # SUM,  sum([batchsize, dim])
        # AUTO/SUM_OVER_BATCH_SIZE, sum([batchsize, dim])/batchsize
        self.bce = BinaryCrossentropy(from_logits=False, \
            reduction=self.reduction)

    def call(self, pred, gt, mask=None, weight=None, name=None):
        loss = self.bce(gt, pred)  # y_true, y_pred
        return loss


class DiceLoss(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def call(self, pred, gt, mask, weights=None):
        # assert pred.shape == gt.shape
        # assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        inter = tf.reduce_sum(pred * gt * mask)
        union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + self.eps
        loss  = 1 - 2.0 * inter/union
        # assert loss <= 1
        return loss


class MaskL1Loss(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(MaskL1Loss, self).__init__()
        self.eps = eps
        
    def call(self, pred, gt, mask):
        loss = tf.reduce_sum(tf.abs(pred - gt) * mask) / (tf.reduce_sum(mask) + self.eps)
        loss = tf.reduce_mean(loss)
        return loss


class BalanceLoss(Layer):
    def __init__(self, \
        balance_loss=True, 
        main_loss_type='DiceLoss', 
        negative_ratio=3, 
        return_origin=False, 
        eps=1e-6, 
        **kwargs):
        super(BalanceLoss, self).__init__()
        
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        elif self.main_loss_type == "Euclidean":
            self.loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction=tf.keras.losses.Reduction.NONE)
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = ["CrossEntropy", "Euclidean", "DiceLoss", "BCELoss", "MaskL1Loss"]
            raise Exception("main_loss_type in BalanceLoss can only be one of {}".format(loss_type))

    def call(self, pred, gt, mask=None):
        # compute main_loss_type
        loss = self.loss(pred, gt, mask=mask)
        print('+++++++', tf.shape(loss), '+++++++')

        if not self.balance_loss:
            return loss

        # compute balance loss
        positive = gt * mask
        negative = (1 - gt) * mask

        #int(positive.sum())
        positive_count = tf.cast(tf.reduce_sum(positive), tf.int32)
        #int(min(negative.sum(), positive_count * self.negative_ratio))
        negative_count = tf.cast(tf.reduce_min(tf.reduce_sum(negative), positive_count * self.negative_ratio), tf.int32)

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = tf.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            # sort_loss = negative_loss.sort(descending=True)
            sort_loss = tf.sort(negative_loss, direction='DESCENDING')
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = paddle.topk(negative_loss, k=negative_count_int)

            # balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            #     positive_count + negative_count + self.eps)
            balance_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
                tf.cast(positive_count, tf.float32) + tf.cast(negative_count, tf.float32) + self.eps)
        else:
            # balance_loss = positive_loss.sum() / (positive_count + self.eps)
            balance_loss = tf.reduce_sum(positive_loss) / (tf.cast(positive_count, tf.float32) + self.eps)

        
        if self.return_origin:
            return balance_loss, loss

        return balance_loss
