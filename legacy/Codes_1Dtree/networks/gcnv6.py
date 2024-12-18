import torch
import torch_scatter
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.resolver import activation_resolver
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import (
    OptTensor,
    Adj
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from typing import Tuple, Union, Optional, Dict, List

_not_none = lambda item: item is not None
##############
class GraphConv(gnn.MessagePassing):
    def __init__(self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, 0)

        self.lin = nn.Sequential()
        self.lin.append(gnn.Linear(2*in_channels[0]+in_channels[1], out_channels, bias=False,
                             weight_initializer='glorot'))
        self.lin.append(nn.LeakyReLU(inplace=True))
        self.lin.append(gnn.Linear(out_channels, out_channels, bias=False,
                             weight_initializer='glorot'))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        # self.lin.reset_parameters()
        gnn.inits.zeros(self.bias)

    def forward(self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor =None):


        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr
        )

        if self.bias is not None:
            out += self.bias
        return out

    def message(self, 
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: OptTensor = None) -> Tensor:
        msg = torch.cat([x_i, x_j], dim=1)
        if edge_attr is not None:
            msg = torch.cat([msg, edge_attr], dim=1)
        return self.lin(msg)
    
    # def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    #     return spmm(adj_t, x, reduce=self.aggr)

    def aggregate(self, edge_attr, edge_index, dim_size=None):
        return torch_scatter.scatter(edge_attr, edge_index[0,:], dim=0,
                                reduce='sum')

##########
def build_multilayers_graphconv(
    in_channels: Union[int, Tuple[int, int]],
    out_channels: int,
    hidden_size: int,
    n_layers: int,
    act: Union[callable, str],
    dropout: float = 0.2
):
    act = activation_resolver(act)
    layers = []

    if isinstance(in_channels, int) or in_channels[1]==0:
        _input_str = 'x, edge_index'
    else:
        _input_str = 'x, edge_index, edge_attr'
    
    layers.append((nn.Dropout(p=dropout), 'x -> x'))
    layers.append((GraphConv(in_channels, hidden_size), _input_str + '->x'))
    layers.append(act)
    for _ in range(n_layers-1):
        layers.append((GraphConv(hidden_size, hidden_size), 'x, edge_index->x'))
        layers.append(act)
    layers.append((GraphConv(hidden_size, out_channels), 'x, edge_index->x'))

    return gnn.Sequential(_input_str,layers)

############
class RecurrentFormulationNet(nn.Module):
    def __init__(self,
        n_fields: Union[int, Tuple[int, int]],
        n_meshfields: Union[int, Tuple[ int, int ]],
        hidden_size: int,
        n_layers: Union[int, Tuple[int, int]],
        act='relu',
        n_previous_timesteps: int=1,
        dropout: float = None,
        use_hidden: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.act = activation_resolver(act)
        self.n_previous_timesteps = n_previous_timesteps

        if isinstance(n_fields, int):
            n_fields = (n_fields, 0)

        if isinstance(n_meshfields, int):
            n_meshfields = (n_meshfields, 0)
        
        if isinstance(n_layers, int):
            n_layers = (n_layers, n_layers)

        # print((n_fields[0]*n_previous_timesteps+n_meshfields[0], n_fields[1]*n_previous_timesteps+n_meshfields[1]))
        # self.encoder = build_multilayers_graphconv(
        #     in_channels=(n_fields[0]*n_previous_timesteps+n_meshfields[0], n_fields[1]*n_previous_timesteps+n_meshfields[1]),
        #     out_channels=hidden_size,
        #     hidden_size=hidden_size,
        #     n_layers=n_layers,
        #     act=act
        # )
        self.encoder = gnn.GraphSAGE(
            in_channels=n_fields[0]*n_previous_timesteps+n_meshfields[0],
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=n_layers[0],
            act=act,
            dropout=dropout
        )

        # self.decoder = nn.LSTM(
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=n_layers[1]
        # )
        self.decoder = gnn.GraphSAGE(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=n_layers[1],
            act=act,
            dropout=dropout
        )

        self.lin = nn.Linear(hidden_size, n_fields[0])
    
    def forward(self, 
        F_0: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Adj, 
        meshfield: Union[Tensor, Tuple[Tensor, Tensor]] = None,
        n_timesteps: int=1,
        time_dim: int=1,
        forward_sequence: bool=False):
        
        if forward_sequence:
            return self.forward_sequence(F_0, edge_index, meshfield, n_timesteps, time_dim)
        else:
            return self.forward_instance(F_0, edge_index, meshfield, n_timesteps, time_dim)

    def forward_sequence(self,
        F_0: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Adj, 
        meshfield: Union[Tensor, Tuple[Tensor, Tensor]] = None,
        n_timesteps: int=1,
        time_dim: int=1):

        # Check input
        if n_timesteps > F_0.size(time_dim) - self.n_previous_timesteps:
            print('Warning: n_timesteps is higher than number of input timesteps!')
            n_timesteps = F_0.size(time_dim) - self.n_previous_timesteps

        # Graph encoder
        if isinstance(F_0, Tensor):
            F_0 = (F_0, None)

        if isinstance(meshfield, Tensor):
            meshfield = (meshfield, None)
        
        x_s = []
        # h_s = []
        h = None

        for i in range(n_timesteps):            

            _range = torch.tensor(list(range(i, i+self.n_previous_timesteps))).to(F_0[0].device)
            F_prev = (
                F_0[0].index_select(time_dim, _range) if F_0[0] is not None else None,
                F_0[1].index_select(time_dim, _range) if F_0[1] is not None else None
            )

            x_prev = F_prev[0][:,-1]

            x = list(filter(_not_none,[F_prev[0].flatten(time_dim,-1), meshfield[0]]))
            x = torch.cat(x, dim=1) if len(x) > 0 else None
            
            edge_attr = list(filter(_not_none,[F_prev[1], meshfield[1]]))
            edge_attr = torch.cat(edge_attr, dim=1) if len(edge_attr) > 0 else None

            if edge_attr is not None:
                x = self.encoder(x, edge_index, edge_attr)
            else:
                x = self.encoder(x, edge_index)

            # Activation
            x = self.act(x)

            # LSTM decoder.
            x = self.decoder(x, edge_index)

            # Activation
            x = self.act(x)

            # Linear
            x = x_prev + self.lin(x)
            x_s.append(x.unsqueeze(time_dim))
            # h_s.append(h)
        
        return torch.cat(x_s, dim=time_dim)
        

    def forward_instance(self,
        F_0: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Adj, 
        meshfield: Union[Tensor, Tuple[Tensor, Tensor]] = None,
        n_timesteps: int=1,
        time_dim: int=1):
        
        # Graph encoder
        if isinstance(F_0, Tensor):
            F_0 = (F_0, None)

        if isinstance(meshfield, Tensor):
            meshfield = (meshfield, None)
        
        x_s = []
        h_s = []
        h = None
        F_prev = F_0

        for _ in range(n_timesteps):

            x_prev = F_prev[0][:,-1]

            x = list(filter(_not_none,[F_prev[0].flatten(time_dim,-1), meshfield[0]]))
            x = torch.cat(x, dim=1) if len(x) > 0 else None
            
            edge_attr = list(filter(_not_none,[F_prev[1], meshfield[1]]))
            edge_attr = torch.cat(edge_attr, dim=1) if len(edge_attr) > 0 else None

            if edge_attr is not None:
                x = self.encoder(x, edge_index, edge_attr)
            else:
                x = self.encoder(x, edge_index)

            # Activation
            x = self.act(x)


            # LSTM decoder.
            x = self.decoder(x, edge_index)

            # Activation
            x = self.act(x)

            # Linear
            x = x_prev + self.lin(x)
            x_s.append(x.unsqueeze(time_dim))
            h_s.append(h)

            # Update F_prev
            temp_F_prev = F_prev[0][:,1:,:]
            temp_F_prev = torch.cat([temp_F_prev, x.detach().unsqueeze(time_dim)], dim=time_dim)
            F_prev = (temp_F_prev, None)
        
        return torch.cat(x_s, dim=time_dim)
    
############################################################################
class GraphNet(nn.Module):
    def __init__(self,
        n_fields: Union[int, Tuple[int, int]],
        n_meshfields: Union[int, Tuple[ int, int ]],
        hidden_size: int,
        n_timesteps: int,
        n_layers: Union[int, Tuple[int, int]],
        act='relu',
        n_previous_timesteps: int=1,
        dropout: float = None,
        use_hidden: bool = True,
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

        # self.net = gnn.GraphSAGE(
        #     in_channels=n_meshfields[0],
        #     hidden_channels=hidden_size,
        #     out_channels=n_fields[0]*n_timesteps,
        #     num_layers=n_layers[0],
        #     act=act,
        #     dropout=dropout
        # )
        # self.net = build_multilayers_graphconv(
        #     in_channels=(n_meshfields[0], 0),
        #     out_channels=n_fields[0]*n_timesteps,
        #     hidden_size=hidden_size,
        #     n_layers=n_layers[0],
        #     act=act
        # )
        self.net = gnn.GraphUNet(
            in_channels=n_meshfields[0],
            hidden_channels=hidden_size,
            out_channels=n_fields[0]*n_timesteps,
            # num_layers=n_layers[0],
            depth=int(n_layers[0] / 2),
            act=act
        )
    
    def forward(self, 
        edge_index: Adj, 
        meshfield: Union[Tensor, Tuple[Tensor, Tensor]] = None
    ):
        if isinstance(meshfield, Tensor):
            meshfield = (meshfield, None)

        x = list(filter(_not_none,[meshfield[0]]))
        x = torch.cat(x, dim=1) if len(x) > 0 else None
        
        edge_attr = list(filter(_not_none,[meshfield[1]]))
        edge_attr = torch.cat(edge_attr, dim=1) if len(edge_attr) > 0 else None

        if edge_attr is not None:
            x = self.net(x, edge_index, edge_attr)
        else:
            x = self.net(x, edge_index)
        return x.view((-1, self.n_timesteps, self.n_fields))