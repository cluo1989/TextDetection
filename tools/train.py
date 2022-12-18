'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-30 10:11:58
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-18 03:16:00
FilePath: /dbnet_plus/tools/train.py
Description: model training 
'''
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard
    )


# GPU setting
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu, \
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1*10240)])

from utils.util import create_module
from utils.util import initial_logger
from utils.parse_config import ArgsParser
from modeling.optimizers.optimizers import WarmUpCosineDecayScheduler

logger = initial_logger()


def train(params):
    # get params
    global_params = params['Global']
    arch_params = params['Architecture']    
    optimizer_params = params['Optimizer']

    # create model and dataset generator
    model = create_module(arch_params['function'])(params)

    # num of replicas
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    logger.info('Number of devices: {}' .format(num_replicas))

    # global batch size
    GLOBAL_BATCH_SIZE = global_params['BATCH_SIZE_PER_REPLICA'] * num_replicas
    
    # build and compile model with distribute strategy
    loss_name = params['Loss']['name']
    optimizer = create_module(optimizer_params['function'])(optimizer_params)

    with strategy.scope():
        model = model('train')
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss={loss_name: lambda y_true, y_pred: y_pred}
        )
    logger.info('build model success!')

    # load pretrained weight
    path = os.path.join(global_params['OUTPUT_DIR'], 'checkpoint')
    files = os.listdir(path)

    load_flag = False
    if len(files) > 0 and global_params['START_EPOCH'] != 0:
        start_weights = "{:04d}".format(global_params['START_EPOCH'])
        for p in files:
            if start_weights in p:
                weight_file = p
                model.load_weights(os.path.join(path, weight_file), by_name=True)
                logger.info('load pretrained weights from: ' + weight_file)
                load_flag = True
                break

    if not load_flag:
        logger.info('train model from begining.')
        
    # callbacks ----
    # warmup & cosin decay
    total_steps = int(global_params['STEPS_PER_EPOCH'] * global_params['EPOCHS'])
    warmup_steps = int(global_params['STEPS_PER_EPOCH'] * global_params['WARMUP_EPOCH'])
    lr_base = float(global_params['WARMUP_LR_BASE']) * num_replicas
    lr_schedule = WarmUpCosineDecayScheduler(\
        learning_rate_base=lr_base,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=0,
        verbose=1
        )

    # early stop
    early_stopping = EarlyStopping(monitor='loss', patience=20)

    # summary
    subdir = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
    logdir = global_params['OUTPUT_DIR'] + 'summary/' + subdir
    logger.info('summary dir:' + logdir)
    tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

    # checkpoint
    ckptdir = global_params['OUTPUT_DIR'] + 'checkpoint/'
    ckptformat = 'weights.{epoch:04d}-{loss:.4f}-{val_loss:.4f}.h5'
    logger.info('checkpoint dir: ' + ckptdir)
    model_checkpoint = ModelCheckpoint(\
        filepath=ckptdir+ckptformat, 
        monitor='val_loss', 
        save_best_only=False, 
        mode='min', 
        save_weights_only=True
        )

    callbacks = [lr_schedule, early_stopping, tensorboard, model_checkpoint]
    # --------------
    # create training dataset
    dataset_params = params['Dataset']
    train_dataset_generator = create_module(dataset_params['function'])(dataset_params)

    train_gen = tf.data.Dataset.from_generator(\
        train_dataset_generator, 
        output_types=(tf.float32, tf.float32), # TODO: update data types
        output_shapes=(tf.TensorShape(global_params['INPUT_SHAPE']),
                       tf.TensorShape((None, None, 5)) # TODO: update data types
                      )
        ).batch(GLOBAL_BATCH_SIZE)

    # TODO: create val dataset

    # start training
    logger.info('start training')
    history = model.fit(\
        tuple(train_gen),
        steps_per_epoch=global_params['STEPS_PER_EPOCH'],
        epochs=global_params['EPOCHS'],
        validation_data=train_gen,
        validation_steps=global_params['VAL_STEPS'],
        validation_freq=global_params['VAL_FREQ'],
        callbacks=callbacks,
        initial_epoch=global_params['START_EPOCH'],
        max_queue_size=20,
        use_multiprocessing=True,
        workers=4
        )
    logger.info('stop training')


if __name__ == "__main__":
    params = ArgsParser().parse_args()
    train(params)
    