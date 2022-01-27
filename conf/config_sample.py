#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class WallRecon:
    # Problem definition
    N_VARS_IN = 2
    N_VARS_OUT = 1
    N_PLANES_OUT = 2
    VARS_NAME_IN = ('u','w','p')
    VARS_NAME_OUT = ('u','v','w')
    
    NORMALIZE_INPUT = False
    SCALE_OUTPUT = False
    
    FLUCTUATIONS_PRED = False
    RELU_THRESHOLD = -1.0
    
    TRAIN_YP = 0
    TARGET_YP = (50)
    
    # Hardware definition
    ON_GPU = True
    N_GPU = 1 
    WHICH_GPU_TRAIN = 0 
    WHICH_GPU_TEST = 1
    # Input for 'CUDA_VISIBLE_DEVICES'

    # Dataset definition
    CUR_PATH = '/storage/FieldRecon'
    DS_PATH = '/storage/Train'
    DS_PATH_TEST = '/storage/Test'

    N_DATASETS = 18
    N_SAMPLES_TRAIN = (1400, 1400, 1400, 1400, 1400, 1400, \
                       1400, 1400, 1400, 1400, 1400, 1400, \
                       1400, 1400, 1400, 1400, 1400, 1400 ) 
                      # list of length N_DATASETS 
    INTERV_TRAIN = 3

    N_SAMPLES_TEST = (10)
    INTERV_TEST = 1 

    TIMESTAMP = '1562346575'
    FROM_CKPT = True
    CKPT = 60
    
    # Prediction that has to be post-processed
    PRED_NB = 0
    
    DATA_AUG = False   # Data augmentation
    
    # Network definition
    NET_MODEL = 1
    # See README for network descriptions
    
    # Model-specific options (syntax: [option]_[net_model])
    PAD_1 = 'wrap' 
    
    # Training options
    INIT = 'random' # default value: 'random'
    INIT_MODEL = '/storage/FieldRecon/.saved_models/' + \
                 'WallRecon1_CNN_uw015_128x128_Re2100.0_lr0.002_decay15drop0.5_relu-1564128335'
    TRANSFER_LEARNING = False
    N_TRAINABLE_LAYERS = 3
    
    N_EPOCHS = 100
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    
    INIT_LR = 0.001
    LR_DROP = 0.5
    LR_EPDROP = 40.0
    
    # Callbacks specifications
    TB_HIST_FREQ = 0
    CKPT_FREQ = 10
    
class OuterRecon:
    # Problem definition
    N_VARS_IN = 3
    N_VARS_OUT = 3
    VARS_NAME = ('u','v','w')
    
    TRAIN_YP = 80
    TARGET_YP = 15
    
    # Hardware definition
    ON_GPU = True
    N_GPU = 1 
    WHICH_GPU_TRAIN = 0
    WHICH_GPU_PRED = 1
    # Input for 'CUDA_VISIBLE_DEVICES'
    
    # Dataset definition
    CUR_PATH = '/storage/Train'
    DS_PATH = '/storage/Test'
    
    N_DATASETS = 1 # fixed for now!
    N_SAMPLES_TRAIN = 25000
    INTERV_TRAIN = 1
    N_SAMPLES_TEST = 25000
    INTERV_TEST = 8
    
    # Network definition
    NET_VERSION = 0
    # See README for network descriptions
    
    # Model-specific options (syntax: [option]_[net_model])
    PAD_1 = 'wrap'       
    
    N_EPOCHS = 20
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    
    INIT_LR = 0.01
    LR_DROP = 0.5
    LR_EPDROP = 50.0
    
    # Callbacks specifications
    TB_HIST_FREQ = 10
    CKPT_FREQ = 10
     
