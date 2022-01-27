#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import time
import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler 

#%% Configuration import
import config

prb_def = os.environ.get('MODEL_CNN', None)

if not prb_def:
    # raise ValueError('"MODEL_CNN" enviroment variable must be defined ("WallRecon" or "OuterRecon")')
    # print('"MODEL_CNN" enviroment variable not defined ("WallRecon" or "OuterRecon"), default value "WallRecon" is used')
    app = config.WallRecon
    prb_def = 'WallRecon'
elif prb_def == 'WallRecon':
    app = config.WallRecon
elif prb_def == 'OuterRecon':
    app = config.OuterRecon
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined either as "WallRecon" or "OuterRecon"')

#%% Training utils

# Credit to Martin Holub for the Class definition
class SubTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    def lr_getter(self):
        # Get vals
        decay = self.model.optimizer.decay
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))

    def on_epoch_end(self, episode, logs = {}):
        logs.update({'learning_rate': self.lr_getter()})
        super(SubTensorBoard, self).on_epoch_end(episode, logs)
        
# Credit to Marcin Możejko for the Callback definition
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        

def step_decay(epoch):
   epochs_drop = app.LR_EPDROP
   initial_lrate = app.INIT_LR
   drop = app.LR_DROP
   lrate = initial_lrate * math.pow(drop,  
           math.floor((epoch)/epochs_drop))
   return lrate

#%% FCN model

from fcn import cnn_model, thres_relu

def get_model(model_config):
    # Callbacks
    tensorboard = SubTensorBoard(
        log_dir='.logs/{}'.format(model_config['name']),
        histogram_freq=app.TB_HIST_FREQ
    )
    
    checkpoint = ModelCheckpoint(
        '.logs/'+model_config['name']+'/model.ckpt.{epoch:04d}.hdf5', 
        verbose=1, period=app.CKPT_FREQ)
    
    lrate = LearningRateScheduler(step_decay)
    time_callback = TimeHistory()
    
    callbacks = [tensorboard, checkpoint, lrate, time_callback]
    
    init_lr = model_config['init_lr']
    
    if model_config['distributed_training']:
       print('Compiling and training the model for multiple GPU') 
       if app.INIT == 'model':
           init_model = tf.keras.models.load_model(model_config['model_path'])
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
       
       with model_config['strategy'].scope():
           CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED)
           
           if app.INIT == 'model':
               print('Weights of the model initialized with another trained model')
               # init_model = tf.keras.models.load_model(model_path)
               # init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
               CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
               os.remove('/tmp/model_weights-CNN_keras_model.h5')
    
               # A smaller learning rate is used in this case
               init_lr = init_lr/2
           
           CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(
                             lr=init_lr))
       
    else:
       CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED) 
       # Initialization of the model for transfer learning, if required
       if app.INIT == 'model':
           print('Weights of the model initialized with another trained model')
           # TODO: check if this condition is still valid for the models that were
           # added later
    #       if int(model_path[-67]) != app.NET_MODEL:
    #           raise ValueError('The model for initialization is different from the model to be initialized')
               
           init_model = tf.keras.models.load_model(model_config['model_path'])
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
           CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
           os.remove('/tmp/model_weights-CNN_keras_model.h5')
           
           # A smaller learning rate is used in this case
           init_lr = init_lr/2
           
           # TODO: Modify this implementation of transfer learning to account for cropping layers
           # if app.TRANSFER_LEARNING == True:
           #     lyrs = CNN_model.layers
           #     n_lyrs = len(lyrs)
           #     for i_l in range(n_lyrs):
           #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)
           #         if i_l <= n_lyrs - (2+3*(app.N_TRAINABLE_LAYERS-1)) - 1:  # Every layer has 3 sublayers (conv+batch_norm+activ), except the last one (no batch_norm)
           #             CNN_model.layers[i_l].trainable = False
           #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)
    
       elif app.INIT == 'random':
           print('Weights of the model initialized from random distributions')
    
       print('Compiling and training the model for single GPU')
       CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(lr=init_lr))
    
        
    print(CNN_model.summary())
        
    return CNN_model, callbacks

def load_trained_model(model_config):
    pred_path = model_config['pred_path']
    init_lr = model_config['init_lr']
    
    if app.FROM_CKPT == True:
        model_path = app.CUR_PATH+'/.logs/'+model_config['name']+'/'
        ckpt = app.CKPT
        init_model = tf.keras.models.load_model(
                model_path+f'model.ckpt.{ckpt:04d}.hdf5',
                custom_objects={"thres_relu": layers.Activation(thres_relu)}
                )
        print('[MODEL LOADING]')
        print('Loading model '+str(app.NET_MODEL)+' from checkpoint '+str(ckpt))    
        pred_path = pred_path+f'ckpt_{ckpt:04d}/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
    else:
        model_path = app.CUR_PATH+'/.saved_models/'
        init_model = tf.keras.models.load_model(
                model_path+model_config['name'],
                custom_objects={"thres_relu": layers.Activation(thres_relu)}
                # custom_objects={"thres_relu": thres_relu}
                )
        print('[MODEL LOADING]')
        print('Loading model '+str(app.NET_MODEL)+' from saved model')
        pred_path = pred_path+'saved_model/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
    
    # If distributed training is used, we need to load only the weights
    if model_config['distributed_training']:
    
       print('Compiling and training the model for multiple GPU')
    
       with model_config['strategy'].scope():
    
           CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED)
    
           CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(lr=init_lr))
    
               
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
           CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
           os.remove('/tmp/model_weights-CNN_keras_model.h5')
           del init_model
               
    
    else:
        CNN_model = init_model
    
        CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(lr=init_lr))
    
            
    print(CNN_model.summary())
    
    return CNN_model
