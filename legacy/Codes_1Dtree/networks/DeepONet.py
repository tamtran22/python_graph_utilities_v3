#IMPORTS#
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GAT

class Net1D(nn.Module):

    def __init__(self,n_branch:int,width:int,depth:int,p:int,act,n_trunk:int=1):
        
        super(Net1D, self).__init__()

        #creating the branch network#
        self.branch_net = Geometric(num_classes=p)
        self.branch_net.float()

        #creating the trunk network#
        self.trunk_net = MLP(input_size=n_trunk,hidden_size=width,num_classes=p,depth=depth,act=act)
        self.trunk_net.float()
        
        self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    
    def forward(self,x_branch,x_branch_index,x_trunk):
        """
            evaluates the operator

            x_branch : input_function
            x_trunk : point evaluating at

            returns a scalar
        """

        # x_branch = self.convert_np_to_tensor(x_branch_)
        # x_trunk = self.convert_np_to_tensor(x_trunk_)
        x_branch = x_branch.to(torch.float32)
        x_trunk = x_trunk.to(torch.float32)
        
        branch_out = self.branch_net.forward(x_branch, x_branch_index)
        trunk_out = self.trunk_net.forward(x_trunk,final_act=True)

        output = branch_out @ trunk_out.t() + self.bias
        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth,act):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        #the activation function#
        self.act = act 

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))
        
    def forward(self, x,final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # No activation after the last layer

        if final_act == False:
            return x
        else:
            return torch.relu(x)
class Geometric(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.geo = GAT(
                    in_channels=13,       # Number of input features per node
                    hidden_channels=256,    # Number of hidden units
                    out_channels=128,       # Number of output classes 
                    num_layers=4,         # Number of GAT layers
                    dropout=0.2,          # Dropout rate
                    heads=16,
                    act="ReLU")           
        self.fc = Linear(128, 256)
        self.out = Linear(256, num_classes)
    def forward(self, x, edge_index):
        x = self.geo(x, edge_index)
        x = self.fc(x)
        x = self.out(x)
        return x