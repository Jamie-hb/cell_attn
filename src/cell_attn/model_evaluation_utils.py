import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

import cell_attn.model_definition_utils as model_definition_utils
from cell_attn.cell_trajectory_data_utils import cellTrajectoriesDataset

# -----------------------------------------------------------------------------

def load_model(stdir: str):
    """ Load a deep attention network model.
        
        Parameters
        --------------
        stdir: str
            Directory where the deep attention network is stored, state_dict should be stored at stdir/model.pth.
            
        Returns
        --------------
        model: model_definition_utils.velocity_predictor
            Deep attention network model.
        
        """
    
    if not os.path.isdir(stdir):
        raise Exception('Given storage directory does not exist.')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load state dictionary.
    model_state = torch.load(os.path.join(stdir, 'model.pth'), weights_only=True, map_location=device)
    
    model = model_definition_utils.__dict__['velocity_predictor']()

    # Load model state
    model.load_state_dict(model_state)

    return model

# -----------------------------------------------------------------------------

def load_data(datadir: str, experiments: list[str] | None =None, batch_size: int =20000):
    """ Loads trajectory input and label data in the form of a DataLoader.
        
        Parameters 
        --------------
        datadir: str
            Directory where input and label data are stored, within this directory should be a set of subdirectories, 
            one for each experiment, each containing a input data file called input_data.npy, and label files 
            called labels.npy, as created by create_data_from_trajectories().
        experiments: list[str] | None, default=None
            Specify which experiments to take data from for the dataset.
        batch_size: int, default=20000
            Batch size of the DataLoader.
            
        Returns
        --------------
        dataloader: torch.utils.data.DataLoader
            DataLoader for trajectory data.
            
        """
        
    dataset = cellTrajectoriesDataset(directory=datadir, experiments=experiments)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader    
    
# -----------------------------------------------------------------------------

def plot_losses(stdir: str, w: float =0.01):
    """ Plot the test and training loss from the training process of a model. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        stdir: str
            Directory where the model (and thus loss files) are stored.
        w: float, default=0.01
            Parameter controlling the degree of smoothing in the plots. A 
            lower w leads to a smoother average. See ewma() for details. 
            
        """
        
    if not os.path.isdir(stdir):
        raise Exception('Given storage directory does not exist.')
        
    losses = pd.read_csv(os.path.join(stdir, 'losses.csv'))
        
    plt.plot([i for i in range(len(losses.train_loss))], losses.train_loss, alpha=0.1, color='blue')        
    plt.plot([i for i in range(len(losses.train_loss))], ewma(losses.train_loss, w), color='blue', label='Training Loss')
    
    plt.plot([i for i in range(len(losses.test_loss))], losses.test_loss, alpha=0.1, color='red')        
    plt.plot([i for i in range(len(losses.test_loss))], ewma(losses.test_loss, w), color='red', label='Test Loss')
    
    plt.xlabel('Batch/100', fontsize=17.5)
        
    plt.legend(fontsize='xx-large')
    
# -----------------------------------------------------------------------------

def plot_losses_per_epoch(stdir: str, w: float =0.01):
    """ Plots the test loss after each epoch. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        stdir: str
            Directory where the model (and thus loss files) are stored.
        w: float, default=0.01
            Parameter controlling the degree of smoothing in the plots. A 
            lower w leads to a smoother average. See ewma() for details. 
            
    """

    if not os.path.isdir(stdir):
        raise Exception('Given storage directory does not exist.')

    losses_per_epoch = pd.read_csv(os.path.join(stdir, 'metrics_per_epoch.csv'))
    test_losses_per_epoch = losses_per_epoch.test_loss

    sns.lineplot(x=[i+1 for i in range(test_losses_per_epoch.shape[0])], y=test_losses_per_epoch)
    plt.xticks(ticks=[i+1 for i in range(test_losses_per_epoch.shape[0])], labels=[str(i+1) for i in range(test_losses_per_epoch.shape[0])])

    plt.ylabel('Test Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)

# -----------------------------------------------------------------------------

def ewma(x, w: float):
    """ Computes an exponential weighted moving average of a time series. Called by plot_losses().

        Parameters
        --------------
        x: 1d array
            Time series data.
        w: float
            Weight to use in ewma calculation. Lower w leads to smoother average. 
            
        Returns
        --------------
        x_smoothed: list
            A smoothed version of x.
            
        """
    
    res = [0]*len(x)
    res[0] = x[0]
    
    for i in range(1, len(x)):
        res[i] = w*x[i] + (1 - w)*res[i-1]
    
    return res

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_attention_heatmap(model, datadir: str | None =None, batch_size: int =20000, dataloader: DataLoader | None =None,
                           experiments: list[str] | None =None,
                           point_size: float =1, min_weight: float | None =None, max_weight: float | None=None,
                           palette: str ='gnuplot', colorbar_ndp: int | None =None):
    """ Plots an attention heatmap for a given trained deep attention network and DataLoader. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        datadir: str, default=None 
            Directory containing input data. Should have subdirectories for each experiment.
        batch_size: int, default=20000 
            How many individual focal & neighbour cell plots to superimpose.
        dataloader: torch.utils.data.DataLoader | None, default=None
            Dataloader to draw data from for plotting. Specifying this overrides datadir, experiments, and batch_size. 
            Negates the need to construct a DataLoader from datadir.
        experiments: list[str] | None, default=None
            If constructing a DataLoader, specify which experiments to take data from.
        point_size: float, default=1
            Size of points in the AHM. Passed to s in seaborn.scatterplot().
        min_weight: float | None, default=None
            Specify the lower weight to normalise the colour palette to. If None, will bet set to the 1st percentile of the
            computed weights.
        max_weight: float | None, default=None
            Specify the upper weight to normalise the colour palette to. If None, will be set to the 99th percentile of the
            computed weights.
        palette: str, default='gnuplot' 
            Specify which matplotlib colour palette to use. 
        colorbar_ndp: int | None, default=None
            Number of decimal places to round color bar labels to. 
            
        """

    if dataloader is None:
        dataset = cellTrajectoriesDataset(directory=datadir, experiments=experiments)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    
    X, y = next(iter(dataloader))

    with torch.no_grad():
        w = model.compute_weights(X).reshape((-1,)).numpy()

    x = X[:,:,6].reshape((-1,))
    y = X[:,:,7].reshape((-1,))  

    # Determine color limits
    if max_weight is None:
        max_weight = np.percentile(w, 99)
    if min_weight is None:
        min_weight = np.percentile(w,1)

    # Make plot
    sns.scatterplot(x=x, y=y, hue=w, palette=palette, hue_norm=(min_weight,max_weight), s=point_size, legend=False)

    # Add color bar
    if colorbar_ndp is None:
        norm = plt.Normalize(min_weight, max_weight)
        ticks = np.linspace(min_weight,max_weight,num=5)
    else:
        norm = plt.Normalize(np.round(min_weight,colorbar_ndp), np.round(max_weight,colorbar_ndp))
        ticks = np.round(np.linspace(min_weight,max_weight,num=5), colorbar_ndp)
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), location='right', pad=0.02, shrink=0.5, extend='both', extendfrac=0.0675, aspect=15, ticks=ticks)
    plt.annotate(text='Normalised\nInteraction\nStrength', xy=(1.02, 0.83), xycoords='axes fraction', fontsize=15)

    # Tidy up axes
    plt.xticks(fontsize=17.5)
    plt.yticks(fontsize=17.5)
    plt.xlabel('')
    plt.ylabel('')

    plt.gca().set_aspect('equal')
    
# -----------------------------------------------------------------------------

def plot_weight_function(model, xlim: tuple[float,float], ylim: tuple[float,float],
                         palette: str ='gnuplot', cutoff_radius: float =None, show_center: bool =True, 
                         min_weight: float =0, max_weight: float | None =None, colorbar_ndp: int | None =None):
    """ Plot the trained weight function of a deep attention network. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        xlim: tuple[float,float]
            Limits for horizontal axis of plot.
        xlim: tuple[float,float]
            Limits for vertical axis of plot.
        pallete: str, default='gnuplot'
            Name of matplotlib colour palette to use.
        cutoff_radius: float, default=None
            If show_center=False, grey out all points inside this radius.
        show_center: bool, default=True
            Specify whether to plot points near the origin. If False, grey out all points
            inside the cutoff_radius.
        min_weight: float, default=0
            Snap all weights below min_weight to min_weight.
        colorbar_ndp: int | None, default=None
            Number of decimal places to round colour bar labels to. 
            
        """

    if (not show_center) & (cutoff_radius==None):
        # NOTE: Currently no provision for having a seperate cutoff for display and for weight computation.
        raise ValueError('If choosing not to show the central circle, must specify cutoff_radius.')
    
    x, y = torch.meshgrid(torch.linspace(xlim[0],xlim[1],501), torch.linspace(ylim[0],ylim[1],501), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    
    with torch.no_grad():
        w = model.weight_network( torch.hstack((x.unsqueeze(-1), y.unsqueeze(-1))).float() ).numpy().flatten()

    x = x.numpy()
    y = y.numpy()
    # Sometimes blowup can occur as x -> focal cell. This stops that distorting the colour map.
    if max_weight==None:
        if not cutoff_radius==None:
            farcell_points = x**2 + y**2 > cutoff_radius**2
            max_weight = np.max(w[farcell_points])
            w[w>max_weight] = max_weight
        else:
            max_weight = np.max(w)

    w[w<min_weight] = min_weight

    if not show_center:
        include_points = x**2 + y**2 > cutoff_radius**2
        notinclude_points = [not b for b in include_points]

    # Make plot
    if show_center:
        sns.scatterplot(x=x, y=y, hue=w, palette=palette, hue_norm=(min_weight, max_weight), s=1, legend=False)
    else:
        sns.scatterplot(x=x[include_points], y=y[include_points], hue=w[include_points], palette=palette, hue_norm=(min_weight, max_weight), s=1, legend=False)
        sns.scatterplot(x=x[notinclude_points], y=y[notinclude_points], color='lightgray', alpha=0.3, s=1, legend=False)

    # Make color bar
    if colorbar_ndp is None:
        norm = plt.Normalize(min_weight, max_weight)
        ticks = np.linspace(min_weight,max_weight,num=5) 
    else:
        norm = plt.Normalize(np.round(min_weight,colorbar_ndp), np.round(max_weight,colorbar_ndp))
        ticks=np.round(np.linspace(min_weight,max_weight,num=5),colorbar_ndp)
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), location='right', pad=0.02, shrink=0.6, extend='both', extendfrac=0.065, aspect=17.5, ticks=ticks)
    plt.annotate(text='Interaction\nStrength', xy=(1.02, 0.9), xycoords='axes fraction', fontsize=15)
    
    plt.gca().set_aspect('equal')

# -----------------------------------------------------------------------------

def plot_attraction_repulsion_heatmap(model, datadir: str | None =None, batch_size: int =20000, 
                                      dataloader: DataLoader | None =None, experiments: list[str] | None =None, 
                                      prediction_timestep: int =1, point_size: float =1, 
                                      palette: str ='seismic_r', colorbar_ndp: int =None):

    """ Plots an attraction-repulsion heatmap for a given trained deep attention network and DataLoader. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        datadir: str, default=None 
            Directory containing input data. Should have subdirectories for each experiment.
        batch_size: int, default=20000 
            How many individual focal & neighbour cell plots to superimpose.
        dataloader: torch.utils.data.DataLoader | None, default=None
            DataLoader to draw data from for plotting. Specifying this overrides datadir, experiments, and batch_size. 
            Negates the need to construct a DataLoader from datadir.
        experiments: list[str] | None, default=None
            Specify which experiments to take data from for the DataLoader, if constructing one.
        prediction_timestep: int, default=1
            Length of the interval over which the deep attention network was trained to predict cell displacement over.
        point_size: float, default=1
            Size of points in the ARHM. Passed to s in seaborn.scatterplot().
        palette: str, default='seismic_r'
            Specify which matplotlib colour palette to use.
        colorbar_ndp: int | None, default=None
            Number of decimal places to round colour bar labels to. 
            
        """

    # Create data loader
    if dataloader is None:
        dataset = cellTrajectoriesDataset(directory=datadir, experiments=experiments)
        dataloader = DataLoader(dataset, batch_size=batch_size)

    X, y = next(iter(dataloader))

    # Nbh cell displacement in focal cell co-ordinates
    x = X[:,:,6].reshape((-1,))
    y = X[:,:,7].reshape((-1,))
    
    nbh_cell_displacements = prediction_timestep * torch.reshape(X[:,:,[0,1]], (X.shape[0]*X.shape[1],2))

    # Create versions with nbh cells removed
    X_drop = torch.zeros((X.shape[0]*X.shape[1], X.shape[1]-1, X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_drop[i*X.shape[1] + j] = torch.concatenate((X[i][:j], X[i][j+1:]), axis=0)

    # Get predictions with one neighbour cell removed
    predictions_drop = model(X_drop)
    del X_drop

    # Get predicted endpoints as usual, and expand to desired shape
    predictions = model(X)
    predictions_all_cells = torch.zeros((X.shape[0]*X.shape[1], 2))
    for i in range(predictions.shape[0]):
        predictions_all_cells[i*X.shape[1]:(i+1)*X.shape[1]] = predictions[i]

    del predictions

    del X

    # Compute proximity to neighbour cell location with the neighbour cell present/removed
    predicted_proximities = ((predictions_all_cells - nbh_cell_displacements)**2).sum(axis=1).sqrt()
    nbh_drop_proximities = ((predictions_drop - nbh_cell_displacements)**2).sum(axis=1).sqrt()
    
    weighted_attraction_scores = 1 * (nbh_drop_proximities - predicted_proximities).detach().numpy()

    # Compute colour palette bounds
    neg_scores = weighted_attraction_scores[weighted_attraction_scores<0]
    pos_scores = weighted_attraction_scores[weighted_attraction_scores>0]
    colour_bound = np.maximum(np.percentile(pos_scores, 99), np.percentile(np.abs(neg_scores), 99))

    # Make plot
    sns.scatterplot(x=x, y=y, hue=weighted_attraction_scores, palette=palette, hue_norm=(-colour_bound,colour_bound), s=point_size, legend=False)

    # Make color bar
    if colorbar_ndp is None:
        norm = plt.Normalize(-colour_bound, colour_bound)
        ticks = np.linspace(-colour_bound, colour_bound,num=5)
    else:
        norm = plt.Normalize(-np.round(colour_bound, colorbar_ndp), np.round(colour_bound, colorbar_ndp))
        ticks = np.round(np.linspace(-colour_bound,colour_bound,num=5), colorbar_ndp)
    ticks[2] = 0
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), location='right', pad=0.02, shrink=0.6, extend='both', extendfrac=0.065, aspect=17.5, ticks=ticks)
    plt.annotate(text='Attraction', xy=(1.02, 0.9), xycoords='axes fraction', fontsize=15)

    # Tidy up axes
    plt.xticks(fontsize=17.5)
    plt.yticks(fontsize=17.5)
    plt.xlabel('')
    plt.ylabel('')

    plt.gca().set_aspect('equal')

# -----------------------------------------------------------------------------

def plot_alignment_heatmap(model, datadir: str | None =None, batch_size: int =20000, 
                           dataloader: DataLoader | None =None, experiments: list[str] | None =None, 
                           point_size: float =1, palette: str ='seismic_r', colorbar_ndp: int | None =None):

    """ Plots an alignment heatmap for a given trained deep attention network and dataloader. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        datadir: str, default=None 
            Directory containing input data. Should have subdirectories for each experiment.
        batch_size: int, default=20000 
            How many individual focal & neighbour cell plots to superimpose.
        dataloader: torch.utils.data.DataLoader | None, default=None
            DataLoader to draw data from for plotting. Specifying this overrides datadir, experiments, and batch_size. 
            Negates the need to construct a DataLoader from datadir.
        experiments: list[str] | None, default=None
            Specify which experiments to take data from for the dataset, if loading in a dataset from datadir.
        point_size: float, default=1
            Size of points in the ARHM. Passed to s in seaborn.scatterplot().
        palette: str, default='seismic_r'
            Specify which matplotlib colour palette to use. 
        colorbar_ndp: int | None, default=None
            Number of decimal places to round colour bar labels to. 
            
        """

    # Create data loader
    if dataloader is None:
        dataset = cellTrajectoriesDataset(directory=datadir, experiments=experiments)
        dataloader = DataLoader(dataset, batch_size=batch_size)

    X, y = next(iter(dataloader))

    # Neighbour cell locations in focal cell co-ordinates
    x = X[:,:,6].reshape((-1,))
    y = X[:,:,7].reshape((-1,))

    neighbour_cell_velocities = torch.reshape(X[:,:,[4,5]], (X.shape[0]*X.shape[1],2))

    # Create versions with nbh cells removed
    X_drop = torch.zeros((X.shape[0]*X.shape[1], X.shape[1]-1, X.shape[2]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_drop[i*X.shape[1] + j] = torch.concatenate((X[i][:j], X[i][j+1:]), axis=0)

    # Get predictions with one neighbour cell removed
    predictions_drop = model(X_drop)
    del X_drop

    # Get predicted endpoints as usual, and expand to desired shape
    predictions = model(X)
    predictions_all_cells = torch.zeros((X.shape[0]*X.shape[1], 2))
    for i in range(predictions.shape[0]):
        predictions_all_cells[i*X.shape[1]:(i+1)*X.shape[1]] = predictions[i]

    del predictions  

    del X

    neighbour_cell_velocities = F.normalize(neighbour_cell_velocities, p=2, dim=1)
    predictions_all_cells = F.normalize(predictions_all_cells, p=2, dim=1)
    predictions_drop = F.normalize(predictions_drop, p=2, dim=1)
    
    # Compute proximity to neighbour cell location with the neighbour cell present/removed
    predicted_alignments = (predictions_all_cells * neighbour_cell_velocities).sum(axis=1)
    nbh_drop_alignments = (predictions_drop * neighbour_cell_velocities).sum(axis=1)
    
    weighted_alignment_scores = 1 * (predicted_alignments - nbh_drop_alignments).detach().numpy()

    # Compute colour palette bounds
    neg_scores = weighted_alignment_scores[weighted_alignment_scores<0]
    pos_scores = weighted_alignment_scores[weighted_alignment_scores>0]
    colour_bound = np.maximum(np.percentile(pos_scores, 99), np.percentile(np.abs(neg_scores), 99))

    # Make plot
    sns.scatterplot(x=x, y=y, hue=weighted_alignment_scores, palette=palette, hue_norm=(-colour_bound,colour_bound), s=point_size, legend=False)

    # Make color bar
    if colorbar_ndp is None:
        norm = plt.Normalize(-colour_bound, colour_bound)
        ticks = np.linspace(-colour_bound, colour_bound,num=5)
    else:
        norm = plt.Normalize(-np.round(colour_bound, colorbar_ndp), np.round(colour_bound, colorbar_ndp))
        ticks = np.round(np.linspace(-colour_bound,colour_bound,num=5), colorbar_ndp)
    ticks[2] = 0
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), location='right', pad=0.02, shrink=0.6, extend='both', extendfrac=0.065, aspect=17.5, ticks=ticks)
    plt.annotate(text='Alignment', xy=(1.02, 0.9), xycoords='axes fraction', fontsize=15)

    # Tidy up axes
    plt.xticks(fontsize=17.5)
    plt.yticks(fontsize=17.5)
    plt.xlabel('')
    plt.ylabel('')

    plt.gca().set_aspect('equal')

# -----------------------------------------------------------------------------

def weight_distance_plot(model, dataloader: DataLoader, plot_averages: bool =True, avg_xlim: tuple[float,float] =None,
                         n_bins: int =100, attraction_scores: bool =False, point_size: float =0.1, 
                         prediction_timestep: int =1):
    """ Plots normalised weight vs distance for one batch of a given DataLoader. Designed to
        be called with an active matplotlib.pyplot.figure.

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        dataloader: torch.data.DataLoader
            Cell trajectory DataLoader.
        plot_averages: bool, default=True
            If True, will plot average weight against distance from neighbour cells.
        avg_xlim: tuple[float,float] | None, default=None
            Specify distance limits to plot average weight between.
        n_bins: int, default=100
            Number of points to plot average weight at. 
        attraction_scores: bool, default=False
            If True, colour points by attraction score.
        point_size: float, default=0.1
            Size of points. Passed to s in seaborn.scatterplot. 
        prediction_timestep: int, default=1
            Timestep over which the deep attention network is trained to predict over.
            
        """
        
    X, y = next(iter(dataloader))
    w = model.compute_weights(X).detach().numpy().flatten()
    d = torch.sqrt(X[:,:,0]**2 + X[:,:,1]**2).detach().numpy().flatten()

    if attraction_scores:
        nbh_cell_displacements = prediction_timestep * torch.reshape(X[:,:,[0,1]], (X.shape[0]*X.shape[1],2))
        
        X_drop = torch.zeros((X.shape[0]*X.shape[1], X.shape[1]-1, X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_drop[i*X.shape[1] + j] = torch.concatenate((X[i][:j], X[i][j+1:]), axis=0)

        # Get predictions with one neighbour cell removed
        predictions_drop = model(X_drop)
        del X_drop

        # Get regular predictions and expand to desired shape
        predictions = model(X)
        predictions_all_cells = torch.zeros((X.shape[0]*X.shape[1], 2))
        for i in range(predictions.shape[0]):
            predictions_all_cells[i*X.shape[1]:(i+1)*X.shape[1]] = predictions[i]

        del predictions

        # Compute attraction scores
        predicted_proximities = ((predictions_all_cells - nbh_cell_displacements)**2).sum(axis=1).sqrt()
        nbh_drop_proximities = ((predictions_drop - nbh_cell_displacements)**2).sum(axis=1).sqrt()
        
        weighted_attraction_scores = 1 * (nbh_drop_proximities - predicted_proximities).detach().numpy()

        # Colour pallete
        neg_scores = weighted_attraction_scores[weighted_attraction_scores<0]
        pos_scores = weighted_attraction_scores[weighted_attraction_scores>0]
        colour_bound = np.maximum(np.percentile(pos_scores, 99), np.percentile(np.abs(neg_scores), 99))

        # Plot
        sns.scatterplot(x=d, y=w, s=point_size, hue=weighted_attraction_scores, palette='seismic_r', hue_norm=(-colour_bound,colour_bound), legend=False)
    
    else:
        sns.scatterplot(x=d, y=w, s=point_size, c='grey')
        
    if plot_averages:
        # plot averages in specified range intersect data range
        if avg_xlim is None:
            avg_xlim = (d.min(), d.max())
        else:
            avg_xlim = (np.max([avg_xlim[0], d.min()]), np.min([avg_xlim[1], d.max()]))
                    
        x_mesh = np.linspace(start=avg_xlim[0], stop=avg_xlim[1], num=n_bins+1)
        a = np.zeros(n_bins)
        a[:] = np.nan
        for i in range(n_bins):
            cells_in_bin = (d>=x_mesh[i]) & (d<x_mesh[i+1])
            if cells_in_bin.any():
                a[i] = w[(d>=x_mesh[i]) & (d<x_mesh[i+1])].mean()

        if attraction_scores:
            sns.lineplot(x=x_mesh[:-1], y=a, c='darkslategray', alpha=0.75, label='Average Weight')
        else:
            sns.lineplot(x=x_mesh[:-1], y=a, c='blue', label='Average Weight')

        plt.legend(fontsize=12.5)
        
# -------------------------------------------------------------

def radial_weight_plot(model, rlim: tuple[float,float], r_meshsize: int =100, theta_meshsize: int =100):
    """ Plots radially averaged weight vs distance for the weight function of a deep atttention network.
        Designed to be called with an active matplotlib.pyplot.figure.
        
        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Trained deep attention network, loaded in using load_model().
        rlim: tuple[float,float]
            Limits to compute average weight between.
        r_meshsize: int, default=100
            Number of points to compute average weight at.
        theta_meshsize: int, default=100 
            Number of angles to use when computing average weight.
            
        """
    
    r_mesh = torch.linspace(rlim[0], rlim[1], r_meshsize)
    theta_mesh = torch.linspace(0, 2*np.pi, theta_meshsize)

    a = np.zeros(len(r_mesh))
    for i, r in enumerate(r_mesh):
        x = r*torch.cos(theta_mesh)
        y = r*torch.sin(theta_mesh)
        with torch.no_grad():
            w = model.weight_network( torch.hstack((x.unsqueeze(-1), y.unsqueeze(-1))).float() ).numpy().flatten()
        a[i] = w.mean()

    sns.lineplot(x=r_mesh, y=a, color='blue')
    