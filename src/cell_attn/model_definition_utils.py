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

