#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import tensorflow as tf
import sys 

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from training_utils import get_model
from tfrecord_utils import get_dataset

#%% Configuration import
import config

prb_def = os.environ.get('MODEL_CNN', None)

if not prb_def:
    print('"MODEL_CNN" enviroment variable not defined ("WallRecon" or "OuterRecon"), default value "WallRecon" is used')
    app = config.WallRecon
    prb_def = 'WallRecon'
elif prb_def == 'WallRecon':
    app = config.WallRecon
elif prb_def == 'OuterRecon':
    app = config.OuterRecon
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined either as "WallRecon" or "OuterRecon"')

os.environ["CUDA_VISIBLE_DEVICES"]=str(app.WHICH_GPU_TRAIN);
#os.environ["CUDA_VISIBLE_DEVICES"]="";

# =============================================================================
#   IMPLEMENTATION WARNINGS
# =============================================================================

# Data augmentation not implemented in this model for now
app.DATA_AUG = False
# Transfer learning not implemented in this model for now
app.TRANSFER_LEARNING = False

#%% Hardware detection and parallelization strategy
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)
print(tf.keras.__version__)

if physical_devices:
  try:
    for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

on_GPU = app.ON_GPU
n_gpus = app.N_GPU

distributed_training = on_GPU == True and n_gpus>1

#%% Dataset and ANN model

tstamp = int(time.time())

dataset_train, dataset_val, n_samp_train, n_samp_valid, model_config = \
    get_dataset(prb_def, app, timestamp=tstamp, 
                train=True, distributed_training=distributed_training)

CNN_model, callbacks = get_model(model_config)

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f"Model name: {model_config['name']}")
print(f'Number of samples for training: {int(n_samp_train)}')
print(f'Number of samples for validation: {int(n_samp_valid)}')
print(f'Total number of samples: {int(n_samp_train+n_samp_valid)}')
print(f"Batch size: {model_config['batch_size']}")
print('')
print(f'Data augmentation: {app.DATA_AUG} (not implemented in this model)')
print(f'Initial distribution of parameters: {app.INIT}')
if app.INIT == 'random':
    print('')
    print('')
if app.INIT == 'model':
    print(f'    Timestamp: {app.INIT_MODEL[-10]}')
    print(f'    Transfer learning: {app.TRANSFER_LEARNING} (not implemented in this model)')
print(f'Prediction of fluctuation only: {app.FLUCTUATIONS_PRED}')
print(f'y- and z-output scaling with the ratio of RMS values : {app.SCALE_OUTPUT}')
print(f'Normalized input: {app.NORMALIZE_INPUT}')
print('')
print('# ====================================================================')

#%% Training and evaluation
train_history = CNN_model.fit(
    dataset_train,
    epochs=app.N_EPOCHS,
    steps_per_epoch=int(np.ceil(n_samp_train/model_config['batch_size'])),
    validation_data=dataset_val,
    validation_steps=int(np.ceil(n_samp_valid/model_config['batch_size'])),
    verbose=2,
    callbacks=callbacks)

# Saving model
save_path = app.CUR_PATH+'/.saved_models/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

tf.keras.models.save_model(
    CNN_model,
    save_path+model_config['name'],
    overwrite=True,
    include_optimizer=True,
    save_format='h5'
)

# Saving history

tLoss = train_history.history['loss']
vLoss = train_history.history['val_loss']
tTrain = callbacks[-1].times

np.savez(save_path+model_config['name']+'_log', 
         tLoss=tLoss, vLoss=vLoss, tTrain=tTrain)

