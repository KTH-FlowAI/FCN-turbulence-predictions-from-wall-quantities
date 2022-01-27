# Convolutional-network models to predict wall-bounded turbulence from wall quantities

## Introduction
The code in this repository features a Python/Tensorflow implementation of a convolutional neural network to the model to predict the two-dimensional velocity-fluctuation fields at different wall-normal locations in a turbulent open channel flow, using the wall-shear-stress components and the wall pressure as inputs. Input data are generated using the pseudo-spectral solver SIMSON. More details about the implementation and the results from the training are available in ["Convolutional-network models to predict wall-bounded turbulence from wall quantities", L. Guastoni, A. Güemes, A.Ianiro, S. Discetti, P. Schlatter, H. Azizpour, R. Vinuesa](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/convolutionalnetwork-models-to-predict-wallbounded-turbulence-from-wall-quantities/3CE4E78C5BAFB370A10BB736A78D3DB6) (2021, *Journal of Fluid Mechanics*)

## Pre-requisites
The code was run successfully using Tensorflow>=2.6.0, using 1 GPU for the training at Re<sub>&tau;</sub>=180 and 2 GPUs at Re<sub>&tau;</sub>=550

## Data
The dataset used for training and testing are available in order to ensure the reproducibility of the results. Please, get in touch using the email address for correspondance in the paper to arrange the transfer.

## Training and inference
The FCN training can be performed after cloning the repository

    git clone https://github.com/KTH-FlowAI/FCN-turbulence-predictions-from-wall-quantities.git
    cd src
    python3 train.py
    
All the training parameters are defined in the [config file](https://github.com/KTH-FlowAI/FCN-turbulence-predictions-from-wall-quantities/blob/master/conf/config_sample.py) (the file needs to be renamed *config.py*)

Inference can be performed as follows:

    cd src
    python3 evaluate.py
    
Inference parameters are also set in the config file.

## Additional information
* Currently, only the FCN implementation is provided. FCN-POD will be added in the future, please get in touch if you are interested in the latter architecture
