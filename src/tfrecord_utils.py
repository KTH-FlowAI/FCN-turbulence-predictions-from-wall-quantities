import tensorflow as tf

import os 
import re
import numpy as np
import functools

#%% TensorFlow functions

@tf.function
def periodic_padding(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    partial_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    left_pad = partial_tensor[:,-padding[1][0]:]
    right_pad = partial_tensor[:,:padding[1][1]]
    
    padded_tensor = tf.concat([left_pad, partial_tensor, right_pad], axis=1)
    
    return padded_tensor

def parser(rec,app,inpt,outpt,target_yp,ypos_Ret,
           pad,pred_fluct,scale_output,avgs,rms,
           norm_input,avgs_in,std_in):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                   allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                   allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                   allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                       allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                       allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, 
                                                       allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # AutoGraph check ---------------------------------------------------------
    # print("Python execution: ", parsed_rec['i_sample'])
    # tf.print("Graph execution: ", parsed_rec['i_sample'])
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    
    if inpt == True:
        padding = tf.cast(pad/2, tf.int32)
        
        nxd = nx + pad
        nzd = nz + pad
        
        # Input processing ----------------------------------------------------
        if norm_input == True:
            inputs = periodic_padding(
                tf.reshape((parsed_rec['comp_raw1']-avgs_in[0])/std_in[0],
                           (nz, nx)),
                          ((padding,padding),(padding,padding)))
        else:
            inputs = periodic_padding(
                tf.reshape(parsed_rec['comp_raw1'],
                          (nz, nx)),
                          ((padding,padding),(padding,padding)))
        inputs = tf.reshape(inputs,(1,nzd,nxd))
        
        for i_comp in range(1,app.N_VARS_IN):
            new_input = parsed_rec[f'comp_raw{i_comp+1}']
            if norm_input == True:
                new_input = (new_input-avgs_in[i_comp])/std_in[i_comp]
            inputs = tf.concat((inputs, 
                tf.reshape(periodic_padding(
                tf.reshape(new_input,(nz, nx)),
                ((padding,padding),(padding,padding))),(1,nzd,nxd))),0)
        
        if outpt == False:
            return inputs
    
    # Output processing -------------------------------------------------------
    nx_out = nx
    nz_out = nz
    
    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nz_out, nx_out))
    
    if pred_fluct == True:    
        output1 = output1 - avgs[0][ypos_Ret[str(target_yp)]]
    
    if app.N_VARS_OUT == 1:
        pass
        outputs = output1
        # return inputs, outputs
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nz_out, nx_out))
        if pred_fluct == True:    
            output2 = output2 - avgs[1][ypos_Ret[str(target_yp)]]
        
        if scale_output == True:
            scaling_coeff2 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / 
                                     rms[1][ypos_Ret[str(target_yp)]], 
                                     tf.float32)
            output2 = output2 * scaling_coeff2
        if app.N_VARS_OUT == 2:
            outputs = (output1, output2)#tf.concat((outputs,output2),0)
            #return inputs, (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nz_out, 
                                                              nx_out))
            if pred_fluct == True:    
                output3 = output3 - avgs[2][ypos_Ret[str(target_yp)]]
            
            if scale_output == True:
                scaling_coeff3 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / 
                                         rms[2][ypos_Ret[str(target_yp)]], 
                                         tf.float32)
                output3 = output3 * scaling_coeff3
            outputs = (output1, output2, output3)#tf.concat((outputs,output2,output3),0)
            #return inputs, (output1, output2, output3)
    if inpt == False:
        return outputs
    else:
        return inputs, outputs

def get_dataset(prb_def, app, timestamp, train=True, distributed_training=True):
    #%% Reading from configuration
    # cur_path = app.CUR_PATH
    if train == True:
        ds_path = app.DS_PATH
    else:
        ds_path = app.DS_PATH_TEST
    
    # Average profiles folder
    avg_path = ds_path +'/.avg/'
    
    if app.NET_MODEL == 1:
        pad = tf.constant(16)
        pad_out = 2
    else:
        pad = tf.constant(0)
        raise ValueError('NET_MODEL = 1 is the only one implentated so far')
    
    if train == True:
        # Number of samples of the training and validation sets
        n_samples = app.N_SAMPLES_TRAIN
        n_samples_tot = np.sum(n_samples)
        if not(type(n_samples) is int):
            print(('WARNING: Assuming that the same number of samples is ',
                   'taken from each file'))
            n_samples = n_samples[0]
        validation_split = app.VAL_SPLIT
        
        tfr_path = ds_path + \
            f'/.tfrecords_singlefile_dt{int(0.45*100*app.INTERV_TRAIN)}_f32/'
        
        epochs = app.N_EPOCHS
    else:
        n_samples = app.N_SAMPLES_TEST
        n_samples_tot = np.sum(n_samples)
        if not(type(n_samples) is int):
            print(('WARNING: Assuming that the same number of samples is ',
                   'taken from each file'))
            n_samples = n_samples[0]
        
        tfr_path = ds_path + \
            f'/.tfrecords_singlefile_test_dt{int(0.45*100*app.INTERV_TEST)}_f32/'
    
    batch_size = app.BATCH_SIZE
    if distributed_training:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.ReductionToOneDevice())
        print('Number of devices for distributed training: ' + 
              '{}'.format(strategy.num_replicas_in_sync))
        print((
        'WARNING: The provided batch size is used in each device ',
        'of the distributed training'))
        batch_size *= strategy.num_replicas_in_sync
    else:
        strategy = None
 
    train_yp = app.TRAIN_YP
    target_yp = app.TARGET_YP
    if not(type(target_yp) is int):
        target_yp = target_yp[0]

    #%% Settings for TFRecords
    tfr_files = [os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if
                 os.path.isfile(os.path.join(tfr_path,f))]
    
    regex = re.compile(f'yp{target_yp}')
    tfr_files = [string for string in tfr_files if re.search(regex, string)]
    
    tfr_files = [string for string in tfr_files if 
                 int(string.split('_')[-2][4:7])<app.N_DATASETS]
    
    # Retrieve simulation parameters from file names
    (nx_, nz_, ny_) = tf.constant([int(val) for val in 
                                   tfr_files[0].split('_')[-5].split('x')])
    
    nx = nx_ + pad
    nz = nz_ + pad
    
    input_shape = (app.N_VARS_IN, nx.numpy(), nz.numpy())
    
    Ret = (tfr_files[0].split('/')[-1]).split('_')[0][3:]
    
    #if len(tfr_files)>1:
    #    # XXX Files are not ordered!
    #    max_samples_per_tfr_fromfile = int(tfr_files[0].split('_')[-2][14:])
    #    max_samples_per_tfr = max_samples_per_tfr_fromfile
    #else:
    print('WARNING: Default value for the number of samples per file')
    if Ret == str(180):   
         max_samples_per_tfr_def = 1500
    elif Ret == str(550):
         max_samples_per_tfr_def = 830
        
    max_samples_per_tfr = max_samples_per_tfr_def
    
    n_samples_per_tfr = np.array([int(s.split('_')[-2][14:]) 
                        for s in tfr_files if int(s.split('_')[-2][4:7])==0])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
    tot_samples_per_ds = sum(n_samples_per_tfr)
                           
    print(
    f'''WARNING: The maximum number of samples per file is fixed to {max_samples_per_tfr}, 
    the number of samples from each SIMSON dataset is {tot_samples_per_ds}, 
    make sure that it corresponds to the actual number of files in the TFRecords''')
    
    if n_samples > tot_samples_per_ds:
        raise ValueError(
    f'''The number of samples per file is too large. 
    It needs to be less or equal to {tot_samples_per_ds}. 
    To load more samples consider loading from multiple files, 
    e.g. N_SAMPLES_TRAIN = (4000, 4000) to load 8000 samples''')
    
    n_tfr_loaded_per_ds = np.sum(
        np.where(cumulative_samples_per_tfr<n_samples,1,0))+1

    tfr_files = [string for string in tfr_files if 
                 int(string.split('_')[-1][:3])<=n_tfr_loaded_per_ds]
    
    if train == True:
        # Separating files for training and validation
        n_samp_train = int(n_samples_tot*(1-validation_split))
        n_samp_valid = n_samples_tot - n_samp_train
          
        (n_files_train, samples_train_left) = np.divmod(n_samp_train,n_samples)
        
        tfr_files_train = [string for string in tfr_files if 
                           int(string.split('_')[-2][4:7])<=n_files_train]
        
        n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr<samples_train_left,1,0))+1
        
        tfr_files_train = [string for string in tfr_files_train if 
                           ((int(string.split('_')[-2][4:7])<n_files_train) or 
                            (int(string.split('_')[-2][4:7])==n_files_train and 
                             int(string.split('_')[-1][:3])<=n_tfr_left))]
        
        tfr_files_train = sorted(tfr_files_train, 
                                 key=lambda s: (int(s.split('_')[-2][4:7]), 
                                                int(s.split('_')[-1][:3])))
        
        if sum([int(s.split('_')[-2][14:]) for s in tfr_files_train]) != n_samp_train:
            shared_tfr = tfr_files_train[-1]
            tfr_files_valid = [shared_tfr]
        else:
            shared_tfr = ''
            tfr_files_valid = list()
        
        tfr_files_valid.extend([string for string in tfr_files if 
                                string not in tfr_files_train])
        
        # Sorting the list of input/output files based on the name 
        # (needed to zip input and output dataset afterwards?)
        tfr_files_valid = sorted(tfr_files_valid, 
                                 key=lambda s: (int(s.split('_')[-2][4:7]), 
                                                int(s.split('_')[-1][:3])))
        
        # File preprocessing with tf.data.Dataset
        shared_tfr_out = tf.constant(shared_tfr)
    else:
        # Sorting the list of input/output files based on the name 
        # (needed to zip input and output dataset afterwards?)
        tfr_files_valid = sorted(tfr_files, 
                                 key=lambda s: (int(s.split('_')[-2][4:7]), 
                                                int(s.split('_')[-1][:3])))
        
    # TF values for the filter functions
    # n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    n_samples_loaded_per_tfr = list()
    if n_tfr_loaded_per_ds>1:
        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(n_samples - \
                                cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])
    else:
        n_samples_loaded_per_tfr.append(n_samples)
    
    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)
    
    if train == True:
        tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
        tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)
        
        if n_tfr_left-1>0:
            samples_train_shared = samples_train_left - \
                cumulative_samples_per_tfr[n_tfr_left-2]
            n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]
        else:
            samples_train_shared = samples_train_left
            n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

        tfr_files_train_ds = tfr_files_train_ds.interleave( 
            lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) \
            if tf.math.equal(x,shared_tfr_out) \
            else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, 
            tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],
            sep='-')[0], tf.int32)-1)),
            cycle_length=16,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Interleaving different TFRecords ---------------------------------------- 
        tfr_files_val_ds = tfr_files_val_ds.interleave(
            lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(
            n_samples_tfr_shared - samples_train_shared) \
            if tf.math.equal(x,shared_tfr_out) \
            else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, 
            tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],
            sep='-')[0], tf.int32)-1)),
            cycle_length=16,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        tfr_files_X_test_ds = tf.data.Dataset.list_files(tfr_files, 
                                                         shuffle=False)
        tfr_files_output_test_ds = tf.data.Dataset.list_files(tfr_files,
                                                         shuffle=False)

        tfr_files_output_test_ds = tfr_files_output_test_ds.interleave(
            lambda x : 
            tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr,
            tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],
            sep='-')[0], tf.int32)-1)),
            cycle_length=1)
        
        tfr_files_X_test_ds = tfr_files_X_test_ds.interleave(
            lambda x : 
            tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr,
            tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],
            sep='-')[0], tf.int32)-1)),
            cycle_length=1)
        
    #%% Loading scaling values for inputs and outputs
    
    # Dictionary for the statistics files from Simson
    ypos_Ret180_129 = {'0':0, '10':20, '15':24, '20':28, '30':34, '50':45,
                       '80':60, '100':69,  '120':78, '150':94}
    ypos_Ret180_65 =  {'0':0, '10':10, '15':12, '20':14, '30':17, '50':23,
                       '80':30, '100':34, '120':39, '150':47}
    ypos_Ret550_193 = {'0':0, '15':20, '30':29, '50':37,
                       '80':48, '100':54, '150':67}
    
    #print('WARNING: the y+ indices are computed only at Re_tau = 180')
    print(Ret, ny_.numpy())
    if Ret == str(180):
        if ny_.numpy() == 129:
            ypos_Ret = ypos_Ret180_129
        elif ny_.numpy() == 65:
            ypos_Ret = ypos_Ret180_65
        else:
            raise ValueError('Wall-normal resolution not supported')
    elif Ret == str(550):
        if ny_.numpy() == 193:
            ypos_Ret = ypos_Ret550_193
        else:
            raise ValueError('Wall-normal resolution not supported')
    
    # Check whether we are predicting the fluctuations
    try:
        pred_fluct = app.FLUCTUATIONS_PRED
        # if pred_fluct == True:
            # NAME = NAME + 'fluct'
        if not(str(target_yp) in ypos_Ret):
            raise ValueError(
    "The selected target does not have a corresponding y-index in simulation")
    except NameError:
        pred_fluct = False
    
    # Check whether inputs are normalized as input Gaussian
    try:
        norm_input = app.NORMALIZE_INPUT
    except NameError:
        norm_input = False
    
    # Checking whether the outputs are scaled with the ratio of RMS values
    try:
        scale_output = app.SCALE_OUTPUT
    except NameError:
        scale_output = False
        
    # Loading the mean profile and the fluctuations intensity if needed
    if pred_fluct == True:
        print('The model outputs are the velocity fluctuations')
        avgs = tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+
                app.VARS_NAME_OUT[0]+'.m').astype(np.float32)[:,1]),(1,-1))
        for i in range(1,app.N_VARS_OUT):
            avgs = tf.concat((avgs, tf.reshape(tf.constant(
                np.loadtxt(avg_path+'mean_'+
                app.VARS_NAME_OUT[i]+'.m').astype(np.float32)[:,1]),(1,-1))),0)
    else:
        avgs = tf.constant(0)
    
    if norm_input == True:
        avg_input_path = ds_path + '/.avg_inputs/'
        print('The inputs are normalized to have a unit Gaussian distribution')
        with np.load(avg_input_path+
                     f'stats_ds{app.N_DATASETS}x{n_samples}'+
                     f'_dt{int(0.45*100*app.INTERV_TRAIN)}.npz') as data: 
            avgs_in = tf.constant(data['mean_inputs'].astype(np.float32)) 
            std_in = tf.constant(data['std_inputs'].astype(np.float32))
    else:
        avgs_in = avgs = tf.constant(0)
        std_in = avgs = tf.constant(0)
    
    if scale_output == True:
        print('The outputs are scaled with the ratio of the RMS values, ',
              'taking the first input as reference')
        rms = tf.reshape(tf.constant(np.loadtxt(avg_path+
                app.VARS_NAME_OUT[0]+'_rms.m')[:,1]),(1,-1))
        for i in range(1,app.N_VARS_OUT):
            rms = tf.concat((rms, tf.reshape(tf.constant(
                np.loadtxt(avg_path+
                app.VARS_NAME_OUT[i]+'_rms.m')[:,1]),(1,-1))),0)
    else:
        rms = avgs = tf.constant(0)
    
    #%% Parsing datasets
    if train == True:
        dataset_parser = tf.function(functools.partial(parser,
                                                app=app,inpt=True,outpt=True,
                                                target_yp=target_yp,
                                                ypos_Ret=ypos_Ret,pad=pad,
                                                pred_fluct=pred_fluct,
                                                scale_output=scale_output,
                                                avgs=avgs,rms=rms,
                                                norm_input=norm_input,
                                                avgs_in=avgs_in,std_in=std_in))
        
        dataset_train = tfr_files_train_ds.map(dataset_parser, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        dataset_val = tfr_files_val_ds.map(dataset_parser, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
       
        ## Datasets size check ---------------------------------------------------------
        #itr = iter(dataset_train)
        #j = 0
        #for i in range(n_samp_train):
        #    example = next(itr)
        #    j += 1
        #
        #try:
        #    example = next(itr)
        #except StopIteration:
        #    print(f'Train set over: {j}')
        #
        #itr1 = iter(dataset_val)
        #jj = 0
        #for i in range(n_samp_valid):
        #    example1 = next(itr1)
        ##    if np.any(np.isnan(example1[0].numpy())):
        ##         sys.exit(1)
        ##     elif np.any(np.isnan(example1[1][0].numpy())):
        ##         sys.exit(2)
        ##     elif np.any(np.isnan(example1[1][1].numpy())):
        ##         sys.exit(3)
        ##     elif np.any(np.isnan(example1[1][2].numpy())):
        ##         sys.exit(4)
        #    
        #    jj += 1
        # 
        #try:
        #    example1 = next(itr1)
        #except StopIteration:
        #    print(f'Valid set over: {jj}')
        #print(NAME)
        #import sys
        #sys.exit(0)

        # Dataset shuffling 
        if Ret == str(180):
            shuffle_buffer = 15000
            n_prefetch = 8
        
            dataset_val = dataset_val.cache()
        elif Ret == str(550): 
            shuffle_buffer = 4200
            n_prefetch = 4
        
        dataset_train = dataset_train.shuffle(shuffle_buffer)
        dataset_train = dataset_train.repeat(epochs)
        dataset_train = dataset_train.batch(batch_size=batch_size)
        dataset_train = dataset_train.prefetch(n_prefetch)

        #dataset_val = dataset_val.shuffle(shuffle_buffer)
        #dataset_val = dataset_val.cache()
        dataset_val = dataset_val.repeat(epochs)
        dataset_val = dataset_val.batch(batch_size=batch_size)
        dataset_val = dataset_val.prefetch(n_prefetch)
    else:
        input_parser = tf.function(functools.partial(parser,
                                                app=app,inpt=True,outpt=False,
                                                target_yp=target_yp,
                                                ypos_Ret=ypos_Ret,pad=pad,
                                                pred_fluct=pred_fluct,
                                                scale_output=scale_output,
                                                avgs=avgs,rms=rms,
                                                norm_input=norm_input,
                                                avgs_in=avgs_in,std_in=std_in))
        
        output_parser = tf.function(functools.partial(parser,
                                                app=app,inpt=False,outpt=True,
                                                target_yp=target_yp,
                                                ypos_Ret=ypos_Ret,pad=pad,
                                                pred_fluct=pred_fluct,
                                                scale_output=scale_output,
                                                avgs=avgs,rms=rms,
                                                norm_input=norm_input,
                                                avgs_in=avgs_in,std_in=std_in))
        
        dataset_test = tfr_files_output_test_ds.map(output_parser, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        X_test = tfr_files_output_test_ds.map(input_parser, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #%% Case name
    
    NAME = prb_def
    # fluctations
    if pred_fluct == True:
            NAME += 'fluct'
    
    NAME += f'{app.NET_MODEL}TF2_{app.N_VARS_IN}'
    
    # Input and output scaling
    if norm_input:
        NAME += 'Norm'
    NAME += f'In-{app.N_VARS_OUT}'
    if scale_output:
        NAME += 'Scaled'
    NAME += 'Out'
    
    # Transfer learning
    # TODO: Update the way the model name is registered in the new name
    # TODO: Add correct naming when the model is loaded from a checkpoint
    if app.INIT == 'model':
        model_path = app.INIT_MODEL
        NAME += '_init' + model_path[-10:]
        if app.TRANSFER_LEARNING == True:
            NAME += 'tr' + str(app.N_TRAINABLE_LAYERS)
    else:
        model_path = ''
 
    # Remaining information
    NAME += f'_{train_yp}'+ \
        str(target_yp) + '_' + str(nx_.numpy()) + \
        'x' + str(nz_.numpy()) +'_Ret'+str(Ret) + \
        '_lr' + str(app.INIT_LR) + '_decay' + \
        str(int(app.LR_EPDROP)) + 'drop'+str(app.LR_DROP) + \
        '_relu-' + str(timestamp)
    
    #%% Model config dictionary
    
    model_config = {'input_shape' : input_shape,
                    'nx_' : nx_,
                    'nz_' : nz_,
                    'padding' : 'valid', # XXX hard-coded value
                    'pad' : pad,
                    'pad_out' : pad_out,
                    'distributed_training' : distributed_training,
                    'strategy' : strategy,
                    'batch_size' : batch_size,
                    'name' : NAME,
                    'init_lr' : app.INIT_LR,
                    'lr_drop' : app.LR_DROP,
                    'lr_epdrop' : app.LR_EPDROP,
                    'histogram_freq' : app.TB_HIST_FREQ,
                    'period' : app.CKPT_FREQ,
                    'model_path' : model_path,
        }
    
    if train == True:
        return dataset_train, dataset_val, n_samp_train, n_samp_valid, \
               model_config
    else:
        model_config['rms'] = rms
        model_config['ypos_Ret'] = ypos_Ret
        
        print('Files used for testing')
        for fl in tfr_files:
            print(fl)
        print('')
        return dataset_test, X_test, n_samples_tot, model_config
