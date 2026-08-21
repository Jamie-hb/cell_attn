import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import os
import pandas as pd
from warnings import warn

import cell_attn.model_definition_utils as model_definition_utils
from cell_attn.cell_trajectory_data_utils import cellTrajectoriesDataset
from cell_attn.training_utils import train_loop

def train_deep_attention_network(datadir: str, stdir: str, 
                                 training_subdirs: list[str] | None =None, validation_subdirs: list[str] | None =None, 
                                 training_speed: str ='auto', batch_size: int | None =None, num_epochs: int | None =None, patience: int | None =None, burn_in: int | None =None, 
                                 learning_rate: list[float] =[0.001, 0.00001], type_aware: bool =False, n_types: int =2):
    """ Trains a deep attention network on a cell trajectory data set. 

        Parameters
        --------------
        datadir: str
            Directory containing the input and label data. Should contain sub-directories named "training_data" 
            and "validation_data", within each are directories containing the input and label data for each experiment.
        stdir: str
            Directory to store trained model and training metrics in. Model state_dict will be stored at stdir/model.pth.
        training_subdirs: list[str] | None, default=None
            Specify to use only certain experiments for training data. Each entry must be a subdirectory in datadir/training_data/. 
        validation_subdirs: list[str] | None, default=None
            Specify to use only certain experiments for validation data. Each entry must be a subdirectory in datadir/validation_data/.
        training_speed: str, default='auto':
            Specify what training behaviour to use. Must be one of 'auto', 'short' or 'long'.
            'auto' sets the number of epochs to the equivalent of 750000 batches of batch size 500. Empirically this is always enough
            to reach a near minimal loss. 'short' and 'long' reduce/increase the number of epochs, patience and burn-in respectively.
            If all of batch_size, num_epochs, patience and burn_in are given, they will override the training_speed argument.
            If some but not all of these are given, then the behaviour specified by training_speed will take precedence.
        batch_size: int | None, default=None
            Batch size to use for training.
            If batch_size, num_epochs, patience and burn_in are given, this will override the training_speed argument.
            If some but not all of these are given, then the behaviour specified by training_speed will take precedence.
        num_epochs: int | None, default=None
            Maximum number of epochs to train for.
            If batch_size, num_epochs, patience and burn_in are given, this will override the training_speed argument.
            If some but not all of these are given, then the behaviour specified by training_speed will take precedence.
        patience: int | None, default=None
            Number of epochs to allow a lack of improvement for before terminating training. 
            To disable early stopping set this equal to num_epochs.
            If batch_size, num_epochs, patience and burn_in are given, this will override the training_speed argument.
            If some but not all of these are given, then the behaviour specified by training_speed will take precedence.
        burn_in: int | None, default=None
            Number of epochs to wait before beginning counting for early stopping.
            If batch_size, patience, patience and burn_in are given, this will override the training_speed argument.
            If some but not all of these are given, then the behaviour specified by training_speed will take precedence.
        learning_rate: list[float], default=[0.001, 0.00001]
            Maximum and minimum learning rate to anneal between.
        type_aware: bool, default=False 
            Specify whether to use a type aware deep attention network.
        n_types: int, default=2 
            If using a type aware deep attention network, specify how many cell types are present in the data.
            
        """

    # Check storage directory exists. Attempt to create it if not.            
    if not os.path.isdir(stdir):
        print('Specified storage directory does not exist. Attempting to create directory.')
        os.makedirs(stdir, exist_ok=True)

    # Check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create training and test data sets
    training_dataset = cellTrajectoriesDataset(directory=os.path.join(datadir, 'training_data'),
                                               experiments=training_subdirs)
    validation_dataset = cellTrajectoriesDataset(directory=os.path.join(datadir, 'validation_data'),
                                               experiments=validation_subdirs)

    # Set training details
    use_set_training_specs = True # flag to sets whether to use manual or pre-specified training specs

    # First check if all of the requried training spec arguments are given. If so, set use_set_training_specs
    # flag to False
    if (not batch_size is None) | (not num_epochs is None) | (not patience is None) | (not burn_in is None):
        if (batch_size is None) | (num_epochs is None) | (patience is None) | (burn_in is None):
            warn('You have specified at least one but not all of batch_size, num_epochs, patience and burn_in. As not all are specified, defaulting to auto training behaviour.')
        else:
            use_set_training_specs = False

    # If not all training specs manually specified, use pre-specified ones
    if use_set_training_specs:
        batch_size = 500
        n_batches_per_epoch = np.max([1, int(len(training_dataset) / batch_size)])
        n_required_batches = 75000 # in all experiments 75000 batches of size 500 was plenty enough to reach minimal loss
        n_required_epochs = int(n_required_batches / n_batches_per_epoch) + 1
        default_patience = np.max([1,int(n_required_epochs / 2)])

        if training_speed=='auto':
            num_epochs = 3*n_required_epochs
            patience = default_patience
            burn_in = np.max([1, n_required_epochs - (patience - 1)])
        elif training_speed=='long':
            num_epochs = 7*n_required_epochs
            patience = 2*default_patience
            burn_in = n_required_epochs
        elif training_speed=='short':
            num_epochs = int(1.5*n_required_epochs)
            patience = int(default_patience / 2) + 1
            burn_in = 1
        else:
            raise ValueError('training_speed must be one of "auto", "long", or "short".') 

    # Create data loaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    # Store training details
    pd.DataFrame({'parameter':['stdir', 'datadir', 'training_subdirs', 'validation_subdirs',
                               'batch_size', 'num_epochs', 'burn_in', 'patience', 'learning_rate', 'type_aware', 'n_types'],
                  'value':[stdir, datadir, training_subdirs, validation_subdirs, batch_size, 
                           num_epochs, burn_in, patience, learning_rate, type_aware,
                           n_types]}).to_csv(os.path.join(stdir, 'model_training_parameters.csv'), index=False)
    
    # Create deep attention network
    if type_aware:
        model = model_definition_utils.__dict__['velocity_predictor_with_types'](n_types=n_types).to(device)
    else:
        model = model_definition_utils.__dict__['velocity_predictor']().to(device)

    # Train deep attention network
    train_loop(model=model, training_dataloader=training_dataloader, test_dataloader=validation_dataloader,
               num_epochs=num_epochs, burn_in=burn_in, patience=patience,
               loss_fn=nn.MSELoss(), learning_rate=learning_rate[0], min_learning_rate=learning_rate[1], stloc=stdir)                                   
