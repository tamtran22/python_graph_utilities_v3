import torch
import torch_scatter
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.resolver import activation_resolver
from torch import Tensor
from torch_geometric.typing import (
    Adj, OptTensor
)
from typing import Union, Tuple

############################################################################
_not_none = lambda item: item is not None
class GraphNet(nn.Module):
    def __init__(self,
        n_fields: Union[int, Tuple[int, int]],
        n_meshfields: Union[int, Tuple[ int, int ]],
        hidden_size: int,
        n_timesteps: int,
        n_layers: Union[int, Tuple[int, int]],
        act='relu',
        n_previous_timesteps: int=1,
        dropout:float=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.act = activation_resolver(act)
        self.n_previous_timesteps = n_previous_timesteps
        self.n_timesteps = n_timesteps
        self.n_fields = n_fields

        if isinstance(n_fields, int):
            n_fields = (n_fields, 0)

        if isinstance(n_meshfields, int):
            n_meshfields = (n_meshfields, 0)
        
        if isinstance(n_layers, int):
            n_layers = (n_layers, n_layers)

        self.net1 = gnn.Sequential('x, edge_index, edge_attr', [
            (GraphConv(n_meshfields, hidden_size), 'x, edge_index, edge_attr -> x'),
            (nn.ReLU(), 'x -> x'),
            (GraphConv(hidden_size, hidden_size), 'x, edge_index-> x'),
        ])
        # self.net2 = gnn.GraphUNet(
        #     in_channels=hidden_size,
        #     hidden_channels=hidden_size,
        #     out_channels=n_fields[0]*n_timesteps,
        #     depth=int(n_layers[0] / 2),
        #     act=self.act,
        # )
        self.net2 = gnn.GraphSAGE(
            in_channels=hidden_size,
            hidden_channels=2*hidden_size,
            num_layers=n_layers[0],
            out_channels=hidden_size,
            dropout=dropout,
            act=self.act
        )
        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_fields[0]*n_timesteps)
        )
    
    def forward(self, data):
        x_temp = self.net1(x=data.node_attr.float(), edge_index=data.edge_index, edge_attr=data.edge_attr.float())
        x = self.act(x_temp)
        x = self.net2(x=x, edge_index=data.edge_index)
        x = self.act(x)
        x = self.net3(x+x_temp)
        x = torch.nn.functional.sigmoid(x)
        return x.view((-1, self.n_timesteps, self.n_fields))

#########################################################################
class GraphConv(gnn.MessagePassing):
    def __init__(self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, 0)

        # self.lin = nn.Sequential()
        # self.lin.append(gnn.Linear(2*in_channels[0]+in_channels[1], out_channels, bias=False,
        #                      weight_initializer='glorot'))
        # self.lin.append(nn.LeakyReLU(inplace=True))
        # self.lin.append(gnn.Linear(out_channels, out_channels, bias=False,
        #                      weight_initializer='glorot'))
        self.node_encode = nn.Linear(in_channels[0], out_channels)
        if in_channels[1] > 0:
            self.edge_encode = nn.Linear(in_channels[1], out_channels)
        else:
            self.register_parameter('edge_encode', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.node_encode.reset_parameters()
        if self.edge_encode is not None:
            self.edge_encode.reset_parameters()
        gnn.inits.zeros(self.bias)

    def forward(self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor =None):

        x = self.node_encode(x)
        if edge_attr is not None:
            edge_attr = self.edge_encode(edge_attr)

        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr
        )

        if self.bias is not None:
            out += self.bias
        return out

    def message(self,
        x_j: Tensor,
        edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j
    
    # def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    #     return spmm(adj_t, x, reduce=self.aggr)

    # def aggregate(self, edge_attr, edge_index, dim_size=None):
    #     return torch_scatter.scatter(edge_attr, edge_index[0,:], dim=0,
    #                             reduce='sum')