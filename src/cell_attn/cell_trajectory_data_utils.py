import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
from warnings import warn
import gc
from itertools import compress
from datetime import datetime

import torch
from torch.utils.data import Dataset



# ===== Data creation =====

def create_data_from_trajectories(X: np.ndarray, nn: int, timestep: int, stdir: str, t_min: int =0, t_max: int | None =None, memory_saver: bool =False):
    """ Takes in cell trajectory data in the form of a trajectory matrix, and creates input and label data ready to
        be trained on by a deep attention network.

        Parameters
        --------------
        X: numpy.ndarray 
            Cell trajectory data matrix (dimension 2 x n_cells x n_timepoints), where X[0,i,t] = x co-ordinate of cell i at time t, and X[1,i,t] = y co-ordinate of cell i at time t.
        nn: int 
            Number of nearest neighbours.
        timestep: int
            Time interval to predict focal cell movement over, measured in indices of the trajectory data matrix.
        stdir: str
            Directory to store input and label data in.
        t_min: int | None, default=0
            Lower limit of time window of trajectory data to use. If given will filter X down to X[:,:,t_min:] before other computations. 
        t_max: int | None, default=None
            Upper limit of time window of trajectory data to use. If given will filter X down to X[:,:,:t_max] before other computations. 
        memory_saver: bool, default=False
            If memory_saver=True, then a more memory efficient (but slower) algorithm will be used. Note this will only use a significantly 
            lower amount of memory if the trajectory data matrix contains a significant amount of missing data. 

        Creates the following files in stdir:
        --------------
        input_data.npy: numpy.ndarray
            Dimension (n_datapoints x n_nbhs x 12) numpy array. input_data[i] is a 2d numpy array with rows corresponding 
            to the nearest neighbour cells of a focal cell, and the columns are (with {num columns})
                nbh_cell_displacement_x _y {2}, focal_cell_velocity_x _y {2}, nbh_cell_velocity_x _y {2},
                nbh_cell_displacement_x _y (fcc) {2}.
            fcc (focal cell coordinates) indicates that the vector is given in coordinates where the y-axis aligns
            with the focal cell's velocity.
        labels.npy: numpy.ndarray:
            Dimension (n_datapoints x 2) numpy array, where row i is the next displacement of the focal cell for input_data[i], measured over the given timestep. 
        
        """

    if not os.path.isdir(stdir):
        print('Specified storage directory does not exist. Attempting to create storage directory.')
        os.makedirs(stdir, exist_ok=True)

    # Log data creation parameters
    pd.DataFrame({'parameter':['n_cells', 'n_its', 'nn', 'timestep', 't_min', 't_max', 'memory_saver', 'date'],
                  'value': [X.shape[1], X.shape[2], nn, timestep, t_min, t_max, memory_saver, '.'.join(str(datetime.now()).split('.')[:-1])]}).to_csv(
        os.path.join(stdir, 'input_data_parameters.csv'), index=False)

    # Filter down to specified time range
    if t_max is None:
        t_max = X.shape[2] # if t_max is not given ensure default behaviour is to not impose an upper time limit
    X = X[:,:,t_min:t_max]

    if memory_saver:
        create_data_from_trajectories_memory_saver(X, nn, timestep, stdir)
    else:
        create_data_from_trajectories_vanilla(X, nn, timestep, stdir)
        

def create_data_from_trajectories_vanilla(X: np.ndarray, nn: int, timestep: int, stdir: str):
    """ Default function to turn a cell trajectory adata matrix into input and label data. Called upon by create_data_from_trajectories(). """

    # Strategy for dealing with missing data: As each "row" of the final data matrix is computed, we first check if the focal cell moves, and after computing the 
    # row we check it does not contain NAs. If either condition fails, the row is added to a list of rows that are removed from the final data matrices at the end.

    if np.isnan(X).sum() > 0.20*X.size:
        warn('X contains a significant amount (' + str(100*np.isnan(X).sum()/X.size) + '%) of missing data, and thus this function maybe using substantially more memory than necessary. If you are experiencing memory capacity issues, consider passing memory_saver=True in create_data_from_trajectories().')
    
    x = X[0,:,:]
    y = X[1,:,:]
    n_cells = x.shape[0]
    n_its = x.shape[1]
    # for each focal cell we have n_its - 2 - timestep number of samples we can compute and test a prediction on
    # as we need focal cell velocity, and location timestep later
    n_valid_timepoints = n_its - 2 - timestep 
    n_datapoints = n_cells * n_valid_timepoints  
    
    data = np.zeros((n_datapoints, nn, 8)) # number of data points x num nbhs x num variables

    # Objects to store labels
    velocities = np.zeros((n_datapoints, 2))

    rows_to_remove = [] # record rows with missing/degenerate data to remove at end
    
    for focal_cell in range(n_cells):
        for t in range(2, n_its - timestep):
            # determine 0th index of data to store info in 
            row = focal_cell*n_valid_timepoints + (t-2)
            
            # determine indices of nearest nbhs
            distances = (x[focal_cell,t] - x[:,t])**2 + (y[focal_cell,t] - y[:,t])**2
            nearest_neighbours = distances.argsort()[1:nn+1]

            # check if focal cell moves
            v = np.array([x[focal_cell,t] - x[focal_cell,t-1], y[focal_cell,t] - y[focal_cell,t-1]])
            if (v**2).sum()!=0:
                v_unit = v / np.sqrt((v**2).sum())
                cob_matrix = np.linalg.inv(np.array([[v_unit[1], v_unit[0]],
                                                     [-v_unit[0],v_unit[1]]]))
                
                # Fill out input data from nearest neighbour cells            
                for k, nbh_cell in enumerate(nearest_neighbours):
                    # nbh cell displacement
                    data[row, k, 0] = x[nbh_cell,t] - x[focal_cell,t]
                    data[row, k ,1] = y[nbh_cell,t] - y[focal_cell,t]
                    # focal cell velocity
                    data[row, k, 2] = v[0]
                    data[row, k, 3] = v[1]
                    # nbh cell velocity
                    data[row, k, 4] = x[nbh_cell,t] - x[nbh_cell,t-1]
                    data[row, k, 5] = y[nbh_cell,t] - y[nbh_cell,t-1]
    
                    # Versions in focal cell co-ordinates
                    # nbh cell displacement 
                    data[row, k, 6:8] = np.matmul(cob_matrix, data[row, k, 0:2])
    
                # Compute velocity label
                velocities[row,0] = x[focal_cell,t+timestep] - x[focal_cell,t]
                velocities[row,1] = y[focal_cell,t+timestep] - y[focal_cell,t]
    
                # CHECK if any of the input data or labels are nan
                if np.isnan(data[row,:,:]).any() | np.isnan(velocities[row,:]).any():
                    rows_to_remove.append(row)
            else:
                # remove entries where focal cell doesn't move
                rows_to_remove.append(row)

    # remove entries where focal cell doesn't move
    data = np.delete(data, rows_to_remove, axis=0)
    velocities = np.delete(velocities, rows_to_remove, axis=0)

    # save results
    np.save(os.path.join(stdir, 'input_data.npy'), data)
    np.save(os.path.join(stdir, 'labels.npy'), velocities)
    
    print('done :)')
    

def create_data_from_trajectories_memory_saver(X: np.ndarray, nn: int, timestep: int, stdir: str):
    """ Alternative function to turn a trajectory matrix into input and label data when the matrix is large and contains
        a large amount of missing data. Called upon by create_data_from_trajectories(). """

    # Strategy for dealing with missing data: Loop through all (focal cell, time) pairs and record which have full cell trajectory data (on both focal cell and nearest neighbour
    # cells), then only compute ML input matrix for these pairs. In this way, the ML data matrix is initialised at the size it ends up as, rather than initialised at the maximum 
    # possible size if there were no missing data then trimmed down later. This can use significantly less memory if the cell trajectory matrix contains a significant amount of 
    # missing data.
    
    n_cells = X.shape[1]
    n_datapoints = 0

    # --- Work out which data to keep ---

    X_bool = np.isnan(X)
    if X_bool.sum() < 0.05*X.size:
        warn('X does not contain much (<5%) missing data, so passing memory_saver=True will not save that much memory.')
    
    # Get times where focal cell has data at times t, t-1, t-2, t+timestep
    X_bool = (~X_bool).all(axis=0)
    
    focal_cell_has_data = X_bool[:,2:-timestep] & X_bool[:,1:-timestep-1] & X_bool[:,0:-timestep-2] & X_bool[:,2+timestep:]

    del X_bool
    gc.collect()

    # Get times where focal cell moves
    focal_cell_moves = ~((np.equal(X[0,:,2:-timestep], X[0,:,1:-timestep-1]) & np.equal(X[1,:,2:-timestep], X[1,:,1:-timestep-1])))

    # Compute matrix indicating which (focal cell, time pairs) have sufficient data    
    cell_time_matrix = np.pad(focal_cell_has_data & focal_cell_moves, ((0,0),(2,timestep)), mode='constant', constant_values=False)

    del focal_cell_moves
    del focal_cell_has_data
    gc.collect()
    
    # Create a list of admissable time points for each focal cell
    cell_time_lists = [ [] ]*n_cells
    for i in range(n_cells):
        cell_time_lists[i] = list(np.argwhere(cell_time_matrix[i,:]).flatten())

    del cell_time_matrix
    gc.collect()
    
    # Go through the admissable (focal cell, time) pairs, removing those where the n-nbh cells do not have data at times t, t-1
    for focal_cell, admissable_times in enumerate(cell_time_lists):
        # Extract times with sufficient focal cell data
        times_to_keep = [True]*len(admissable_times)
    
        # Get distances to all nbh cells at all admissable times
        D = ((X[:,:,admissable_times] - np.expand_dims(X[:,focal_cell,admissable_times], axis=1))**2).sum(axis=0)

        nn_idx = D.argsort(axis=0)[1:nn+1,:]
        # For each time point, get n-nbh data and see whether it is all present
        for i, t in enumerate(admissable_times):
            if np.isnan(D[nn_idx[-1,i],i]):
                times_to_keep[i] = False
            else:
                X_nn = X[:,nn_idx[:,i],:][:,:,[t-1,t]]
                # check none of it is na
                if np.isnan(X_nn).any():
                    times_to_keep[i] = False
                    
        # Update cell_time_lists and n_datapoints
        cell_time_lists[focal_cell] = list(compress(admissable_times, times_to_keep))
        n_datapoints += sum(times_to_keep)

    del D
    del nn_idx
    gc.collect()


    # --- Create data matrix ---
    
    data = np.zeros((n_datapoints, nn, 8))
    velocities = np.zeros((n_datapoints, 2))
    
    row = 0 # counter for which row to add data to
    for focal_cell in range(n_cells):
        for t in cell_time_lists[focal_cell]:
            
            focal_cell_data = X[:,focal_cell,[t-2,t-1,t,t+timestep]]
            
            # determine indices of nearest nbhs
            distances = (focal_cell_data[0,2] - X[0,:,t])**2 + (focal_cell_data[1,2] - X[1,:,t])**2
            nearest_neighbours = distances.argsort()[1:nn+1] # NOTE: argsort() treats nan as infinitely large, so this code should work unless there are <nn non-na nbhs
            
            v = np.array([focal_cell_data[0,2] - focal_cell_data[0,1],
                          focal_cell_data[1,2] - focal_cell_data[1,1]])
            
            v_unit = v / np.sqrt((v**2).sum())
            cob_matrix = np.linalg.inv(np.array([[v_unit[1], v_unit[0]],
                                                 [-v_unit[0],v_unit[1]]]))
        
            # Fill out input data from nearest neighbour cells 
            nbh_cell_data = X[:,nearest_neighbours,:][:,:,[t-1,t]]
            for k in range(nn):
                # nbh cell displacement
                data[row, k, 0] = nbh_cell_data[0,k,1] - focal_cell_data[0,2]
                data[row, k ,1] = nbh_cell_data[1,k,1] - focal_cell_data[1,2]
                # focal cell velocity
                data[row, k, 2] = v[0]
                data[row, k, 3] = v[1]
                # nbh cell velocity
                data[row, k, 4] = nbh_cell_data[0,k,1] - nbh_cell_data[0,k,0]
                data[row, k, 5] = nbh_cell_data[1,k,1] - nbh_cell_data[1,k,0]
                
                # Versions in focal cell co-ordinates
                # nbh cell displacement 
                data[row, k, 6:8] = np.matmul(cob_matrix, data[row, k, 0:2])

            # Compute velocity label
            velocities[row,0] = focal_cell_data[0,3] - focal_cell_data[0,2]
            velocities[row,1] = focal_cell_data[1,3] - focal_cell_data[1,2]

            row += 1

    del X
    del cell_time_lists
    gc.collect()
    
    # save results
    np.save(os.path.join(stdir, 'input_data.npy'), data)
    np.save(os.path.join(stdir, 'labels.npy'), velocities)

    print('done :)')


# ===== Dataset definition =====

class cellTrajectoriesDataset(Dataset):
    """ Dataset class for cell trajectory data. 

        Parameters
        --------------    
        directory: str 
            Directory where input and label data is stored, within this directory should be a set of subdirectories, one for each experiment, each containing
            a input data file called input_data.npy, and label files called labels.npy, as created by create_data_from_trajectories().
        experiments: list[str] | None, default=None
            Specify which experiments to take data from for the dataset. Each element of the list should be the name of a subdirectory of directory.
        nsamples_per_run: int | None, default=None
            Only include a fixed number of data points per experiment (chosen uniformly at random).
        device: torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            What device to put the dataset on. 
        
        """
    
    def __init__(self, directory, experiments=None, nsamples_per_run=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        # Check directory, and if given subdirectories in experiments, exist. 
        if not os.path.isdir(directory):
            raise Exception('Specified data directory does not exist.')

        if not experiments is None:
            num_existing_experiments = 0
            for experiment in experiments:
                if not os.path.isdir(os.path.join(directory, experiment)):
                    warn('Warning: subdirectory for experiment '+experiment+' does not exist. Creating Dataset from all experiments with existing subdirectories')
                else:
                    num_existing_experiments += 1

            if num_existing_experiments==0:
                raise Exception('None of the specified subdirectories exist. Unable to create Dataset with no data.')
            
        data_frames = []
        labels = []
        
        # Get folder for each experiment, avoiding hidden directories
        if not experiments is None:
            subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory,subdir)) & (subdir[0]!='.') & (subdir in experiments)]
        else:
            subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory,subdir)) & (subdir[0]!='.')]
        
        if not nsamples_per_run==None:
            rng = np.random.default_rng()
            
        for subdirectory in subdirectories:
            x = np.load(os.path.join(directory, subdirectory, 'input_data.npy'))
            y = np.load(os.path.join(directory, subdirectory, 'labels.npy'))
            
            if not nsamples_per_run==None:
                idx = rng.choice(range(x.shape[0]), nsamples_per_run, replace=False)
                data_frames.append(x[idx,:,:])
                labels.append(y[idx])
            else:
                data_frames.append(x)
                labels.append(y)
                
        self.trajectories = np.concatenate(data_frames)
        self.labels = np.concatenate(labels)
        
        self.device = device
            
    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, i):
        input_data = torch.tensor(self.trajectories[i]).float().to(self.device)
        label = torch.tensor(self.labels[i]).float().to(self.device)

        return input_data, label
    