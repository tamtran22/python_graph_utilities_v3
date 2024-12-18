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

def GCNBlock(in_channels, out_channels, depth=1, act='relu', norm=True, dropout=None):
    _layers = [(gnn.GCNConv(in_channels, out_channels), 'x, edge_index, edge_weight -> x')]
    for _ in range(depth - 1):
        _layers.append(((activation_resolver(act)), 'x -> x'))
        _layers.append((gnn.GCNConv(out_channels, out_channels), 'x, edge_index, edge_weight -> x'))
    if norm:
        _layers.append((gnn.InstanceNorm(out_channels), 'x -> x'))
    if dropout is not None:
        _layers.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))
    return gnn.Sequential('x, edge_index, edge_weight', _layers)


def GPLBlock(in_channels, out_channels, depth=1, act='relu', edge_channels=0, norm=True, dropout=None):
    _layers = [(ProcessorLayer(in_channels, out_channels, hidden_channels=out_channels, n_hiddens=depth, 
                                act=act, edge_channels=edge_channels), 'x, edge_index -> x')]
    for _ in range(depth - 1):
        _layers.append(((activation_resolver(act)), 'x -> x'))
        _layers.append((ProcessorLayer(out_channels, out_channels, hidden_channels=out_channels, n_hiddens=depth, 
                                       act=act, edge_channels=edge_channels), 'x, edge_index -> x'))
    if norm:
        _layers.append((gnn.InstanceNorm(out_channels), 'x -> x'))
    if dropout is not None:
        _layers.append((nn.Dropout(p=dropout, inplace=True), 'x -> x'))
    return gnn.Sequential('x, edge_index', _layers)

###################################################################################

class GraphUNet(nn.Module):
    def __init__(self, 
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        block_depth: int = 1,
        dropout: float = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert depth > 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.sum_res = sum_res
        self.act = activation_resolver(act)
        ##
        self.input = gnn.Linear(in_channels, hidden_channels, bias=True, weight_initializer='glorot')
        self.output = gnn.Linear(hidden_channels+(not sum_res)*hidden_channels, out_channels, bias=True, weight_initializer='glorot')
        self.latent = GPLBlock((2**depth)*hidden_channels, (2**depth)*hidden_channels, depth=2*block_depth,
                                act=act, norm=False, dropout=dropout)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth):
            upper_channels = (2**i)*hidden_channels
            lower_channels = (2**(i+1))*hidden_channels
            # self.downs.append(nn.Linear(in_features=upper_channels, out_features=lower_channels))
            # self.ups.append(nn.Linear(in_features=lower_channels+(not sum_res)*lower_channels- \
            #                           (not sum_res)*(i==depth-1)*lower_channels, out_features=upper_channels))
            self.downs.append(GPLBlock(in_channels=upper_channels, out_channels=lower_channels, 
                                        depth=block_depth, act=act, norm=False, dropout=dropout))
            self.ups.append(GPLBlock(in_channels=lower_channels+(not sum_res)*lower_channels- \
                                        (not sum_res)*(i==depth-1)*lower_channels, out_channels=upper_channels, \
                                        depth=block_depth, act=act, norm=False, dropout=dropout))
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: OptTensor = None) -> Tensor:
        x = self.input(x)
        x = self.act(x)
        x_downs = []
        for i in range(self.depth):
            x_downs.append(x)
            x = self.downs[i](x, edge_index)
            x = self.act(x)
        x = self.latent(x, edge_index)
        for i in reversed(range(self.depth)):
            x = self.ups[i](x, edge_index)
            x = (x + x_downs[i]) if self.sum_res else torch.cat([x, x_downs[i]], dim=-1)
            x = self.act(x)
        x = self.output(x)
        return x


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

        self.mesh_decriptor = GraphUNet(
            in_channels=n_meshfield,
            hidden_channels=hidden_size,
            out_channels=latent_size,
            depth=4,
            block_depth=1,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator1 = GraphUNet(
            in_channels=n_field+latent_size+use_time_feature+use_hidden*hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            depth=4,
            block_depth=1,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator2 = nn.Linear(hidden_size, n_field)

        self.integrator = gnn.MLP(
            in_channels=n_field*2,
            hidden_channels=hidden_size,
            out_channels=n_field,
            num_layers=5,
            act=act
        )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1, device=None):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        # print(meshfield)
        # dt = 4.0/200
        ##
        F_dots, Fs = [], []
        F_current = F_0
        F_hidden = torch.zeros((F_0.size(0), self.hidden_size)).float().to(device)
        for i in range(n_time):
            if time is None:
                x = torch.cat([F_current, meshfield], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                x = torch.cat([F_current, meshfield, _time_current], dim=1)
            if self.use_hidden:
                x = torch.cat([x, F_hidden], dim=1)
            F_hidden = self.act(self.differentiator1(x, edge_index))
            F_dot_current = F.tanh(self.differentiator2(F_hidden))

            x = torch.cat([F_current, F_dot_current], dim=-1)
            F_next = self.integrator(x, edge_index)
            F_next = F.tanh(F_next)
            # F_next = F.tanh(F_current + F_dot_current*dt)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)