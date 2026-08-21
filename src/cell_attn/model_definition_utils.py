""" Model definitions for Cell Movement Attention Networks """

import torch
import torch.nn.functional as F
from torch import nn

# ========== Models ==========

class velocity_predictor(nn.Module):
    """ End to end function for predicting the next displacement of a focal cell. """
    
    def __init__(self):
        super().__init__()
        self.interaction_network = pairwise_function_velocity()
        self.weight_network = weight_function()

    def forward(self, x):
        # Π acts on the nbh cell displacement, focal & nbh cell velocities in Euclidean co-ordinates
        # this is columns 0,1,...,5 of the input data
        interactions = self.interaction_network(x[:,:,:6])
        # W acts on the nbh cell position in focal cell co-ordinate frame
        # this is columns 6,7 in the input data
        weights = self.weight_network(x[:,:,6:8])
        weights = F.normalize(weights, p=1)

        res = torch.sum( torch.mul(interactions, weights), dim=1 )

        return res

    def compute_weights(self, x):
        if x.ndim==2:
            raise TypeError('Warning: compute_weights() is designed to operate on a batch, do not pass it a single sample')
        weights = self.weight_network(x[:,:,6:8])
        res = F.normalize(weights, p=1)

        return res

# -------------------------------------------------------------

class velocity_predictor_with_types(nn.Module):
    """ End to end function for predicting the next displacement of a focal cell, when given cell type data.
    
        Parameters
        --------------
        n_types: int, default=2
            Number of cell types in the data.
        
        """
    
    def __init__(self, n_types: int =2):
        super().__init__()
        self.interaction_network = pairwise_function_velocity_with_types(n_types=n_types)
        self.weight_network = weight_function_with_types(n_types=n_types)
        self.n_types = n_types

    def forward(self, x):
        # Π acts on the nbh cell displacement, focal & nbh cell velocities in Euclidean co-ordinates, and cell type information
        # this is columns 0,1,...,5 and 8,...,7+2*n_types of the input data
        interactions = self.interaction_network( torch.concatenate((x[:,:,:6], x[:,:,8:8+2*self.n_types]), axis=-1) )
        # W acts on the nbh cell position in focal cell co-ordinate frame, and cell type information
        # this is columns 6,7 and 8,...,7+2*n_types in the input data
        weights = self.weight_network( torch.concatenate((x[:,:,6:8], x[:,:,8:8+2*self.n_types]), axis=-1) )
        weights = F.normalize(weights, p=1)

        res = torch.sum( torch.mul(interactions, weights), dim=1 )

        return res

    def compute_weights(self, x):
        if x.ndim==2:
            raise TypeError('Warning: compute_weights() is designed to operate on a batch, do not pass it a single sample')
        weights = self.weight_network( torch.concatenate((x[:,:,6:8], x[:,:,8:8+2*self.n_types]), axis=-1) )
        res = F.normalize(weights, p=1)

        return res


# ===== MLP Architectures =====

class pairwise_function_velocity(nn.Module):
    """ Pairwise interaction function definition.
        
        Input: focal cell velocity (2d), nbh cell relative position (2d) and velocity (2d). 
        
        """
    
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(6,128)
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,2)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# -------------------------------------------------------------

class pairwise_function_velocity_with_types(nn.Module):
    """ Type aware pairwise interaction function definition.
    
        Input: focal cell velocity (2d), nbh cell relative position (2d) and velocity (2d), 
               focal cell and nbh cell type (2*n_types d). 

        Parameters
        --------------
        n_types: int, default=2
            Number of cell types in the data.
        
        """
    
    def __init__(self, n_types: int =2):
        super().__init__()
        self.fc0 = nn.Linear(6+2*n_types,256)
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,2)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# -------------------------------------------------------------

class weight_function(nn.Module):
    """ Weight function definition.
    
        Input: nbh cell relative position (2d). 
        
        """

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(2,128)
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.exp(self.fc3(x))

        return x

# -------------------------------------------------------------

class weight_function_with_types(nn.Module):
    """ Type aware weight function definition.
    
        Input: nbh cell relative position (2d), focal cell and nbh cell type (2*n_types d).
        
        Parameters
        --------------
        n_types: int, default=2
            Number of cell types in the data.
        
        """

    def __init__(self, n_types: int =2):
        super().__init__()
        self.fc0 = nn.Linear(2+2*n_types,256)
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.exp(self.fc4(x))

        return x

