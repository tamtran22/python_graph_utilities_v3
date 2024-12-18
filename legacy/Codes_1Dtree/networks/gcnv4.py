import torch
import torch_geometric.nn as gnn
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union, Callable
from torch import Tensor
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat
import torch.nn.functional as F
import torch_scatter

##################################################################################

class ProcessorLayer(gnn.MessagePassing):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_hiddens: int,
        edge_channels: int=0,
        aggr='sum',
        act='relu',
        **kwargs
    ):
        super().__init__(aggr, **kwargs)
        self.edge_mlp = gnn.MLP(
            in_channels=2*in_channels+edge_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=n_hiddens,
            act=act
        )
    def forward(self, x, edge_index, edge_attr: OptTensor=None, size=None):
        out, updated_edges = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size
        )
        return out
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor=None) -> Tensor:
        if edge_attr is None:
            updated_edges = torch.cat([x_i, x_j], dim=-1)
        else:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            updated_edges = torch.cat([x_i, x_j, edge_attr], dim=-1)
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    def aggregate(self, updated_edges, edge_index, dim_size=None):
        out = torch_scatter.scatter(updated_edges, edge_index[0,:], dim=0, reduce=self.aggr)
        return out, updated_edges


##################################################################################

def GCNBlock(in_channels, out_channels, hidden_channels, depth=1, act='relu', norm=True, dropout=None):
    _layers = [(gnn.GCNConv(in_channels, hidden_channels), 'x, edge_index -> x')]
    if norm:
        _layers.append((gnn.InstanceNorm(hidden_channels), 'x -> x'))
    for _ in range(depth - 2):
        _layers.append(((activation_resolver(act)), 'x -> x'))
        _layers.append((gnn.GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'))
    if dropout is not None:
        _layers.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))
    _layers.append((gnn.GCNConv(hidden_channels, out_channels), 'x, edge_index -> x'))
    return gnn.Sequential('x, edge_index', _layers)


def GPLBlock(in_channels, out_channels, hidden_channels, depth=1, act='relu', edge_channels=0, norm=True, dropout=None):
    _layers = [(ProcessorLayer(in_channels, hidden_channels, hidden_channels=out_channels, n_hiddens=depth, 
                                act=act, edge_channels=edge_channels), 'x, edge_index -> x')]
    if norm:
        _layers.append((gnn.InstanceNorm(hidden_channels), 'x -> x'))
    for _ in range(depth - 2):
        _layers.append(((activation_resolver(act)), 'x -> x'))
        _layers.append((ProcessorLayer(hidden_channels, hidden_channels, hidden_channels=out_channels, n_hiddens=depth, 
                                       act=act, edge_channels=edge_channels), 'x, edge_index -> x'))
    if dropout is not None:
        _layers.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))
    _layers.append((gnn.GCNConv(hidden_channels, out_channels), 'x, edge_index -> x'))
    return gnn.Sequential('x, edge_index', _layers)

###################################################################################

class RecurrentFormulationNet(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: int,
        hidden_size: int,
        latent_size: int,
        act='relu',
        use_time_feature: bool = False,
        dropout: float = None,
        use_hidden: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.use_hidden = use_hidden
        self.act = activation_resolver(act)

        self.mesh_decriptor = GCNBlock(
            in_channels = n_meshfield, 
            out_channels = hidden_size,
            hidden_channels=hidden_size, 
            depth=10, 
            act=act, 
            norm=True, 
            dropout=dropout
        )

        # self.mesh_decriptor = GraphUNet(
        #     in_channels=n_meshfield,
        #     hidden_channels=hidden_size,
        #     out_channels=latent_size,
        #     depth=4,
        #     block_depth=1,
        #     sum_res=False,
        #     act=act,
        #     dropout=dropout
        # )

        self.differentiator = GCNBlock(
            in_channels = n_field+latent_size+use_time_feature, #+use_hidden*hidden_size, 
            out_channels = n_field,
            hidden_channels=hidden_size,
            depth=10, 
            act=act, 
            norm=True, 
            dropout=dropout
        )
        # self.differentiator1 = GraphUNet(
        #     in_channels=n_field+latent_size+use_time_feature+use_hidden*hidden_size,
        #     hidden_channels=hidden_size,
        #     out_channels=hidden_size,a
        #     depth=4,
        #     block_depth=1,
        #     sum_res=False,
        #     act=act,
        #     dropout=dropout
        # )

        # self.differentiator2 = nn.Linear(hidden_size, n_field)

        # self.integrator = GCNBlock(
        #     in_channels=n_field*2,
        #     out_channels=n_field,
        #     hidden_channels=hidden_size,
        #     depth=10,
        #     act=act,
        #     norm=True,
        #     dropout=dropout
        # )
        # self.integrator = gnn.MLP(
        #     in_channels=n_field*2,
        #     hidden_channels=hidden_size,
        #     out_channels=n_field,
        #     num_layers=5,
        #     act=act
        # )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1, forward_sequence=False, device=None):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        dt = 5*(4.0/200)

        ##
        F_dots, Fs = [], []
        if not forward_sequence:
            F_current = F_0
        # F_hidden = torch.zeros((F_0.size(0), self.hidden_size)).float().to(device)
        for i in range(n_time):
            if forward_sequence:
                F_current = F_0[:, i]
            if time is None:
                x = torch.cat([F_current, meshfield], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                x = torch.cat([F_current, meshfield, _time_current], dim=1)
            # if self.use_hidden:
            #     x = torch.cat([x, F_hidden], dim=1)
            F_hidden = self.differentiator(x, edge_index)
            F_dot_current = F.tanh(F_hidden)


            # x = torch.cat([F_current, F_dot_current], dim=-1)
            # F_next = self.integrator(x, edge_index)
            # F_next = F.tanh(F_next)

            F_next = F.tanh(F_current + F_dot_current*dt)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            if not forward_sequence:
                F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)
    
#########################
class RecurrentFormulationNet1(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: int,
        hidden_size: int,
        latent_size: int,
        input_time:int = 1,
        act='relu',
        use_time_feature: bool = False,
        dropout: float = None,
        use_hidden: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.use_hidden = use_hidden
        self.act = activation_resolver(act)
        self.input_time = input_time

        self.mesh_decriptor = GCNBlock(
            in_channels = n_meshfield, 
            out_channels = hidden_size,
            hidden_channels=hidden_size, 
            depth=10, 
            act=act, 
            norm=True, 
            dropout=dropout
        )

        self.differentiator = GCNBlock(
            in_channels = n_field*input_time+latent_size+use_time_feature, #+use_hidden*hidden_size, 
            out_channels = n_field,
            hidden_channels=hidden_size,
            depth=10, 
            act=act, 
            norm=True, 
            dropout=dropout
        )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1, device=None):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        dt = 5*(4.0/200)

        ##
        F_dots, Fs = [], []
        F_input = [F_0[:,i] for i in range(self.input_time)]
        # F_hidden = torch.zeros((F_0.size(0), self.hidden_size)).float().to(device)
        for i in range(n_time):
            if time is None:
                x = torch.cat(F_input + [meshfield], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                # print(F_input[0].size(), meshfield.size(), _time_current.size())
                x = torch.cat(F_input + [meshfield, _time_current], dim=1)
            F_hidden = self.differentiator(x, edge_index)
            F_dot_current = F.tanh(F_hidden)

            F_next = F.tanh(F_input[-1] + F_dot_current*dt)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            
            F_input.pop()
            F_input.append(F_next.detach())
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)