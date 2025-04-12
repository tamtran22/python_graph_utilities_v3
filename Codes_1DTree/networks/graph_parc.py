import torch
# import torch_scatter
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.resolver import activation_resolver
from torch import Tensor
from torch_geometric.typing import (
    Adj, OptTensor
)
from typing import Union, Tuple, Callable
from torch_geometric.nn import SAGEConv
from neuralop.models.fno import FNO

class GraphPARC(nn.Module):
    def __init__(self,
        n_fields: int,
        n_meshfields: Union[int, Tuple[int, int]],
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        act: Union[str, Callable] = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.act = activation_resolver(act)

        if isinstance(n_meshfields, int):
            n_meshfields = (n_meshfields, 0)
        
        _differentiator_in_channels = n_fields + n_meshfields[0]
        _differentiator_out_channels = n_fields
        self.differentiator = gnn.Sequential( 'x, edge_index',[
            (SAGEConv(_differentiator_in_channels, hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            # (gnn.LayerNorm(hidden_channels), 'x->x'),
            (nn.Dropout(dropout), 'x->x'),
            (self.act, 'x->x'),
            (SAGEConv(hidden_channels, 2*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(2*hidden_channels, 4*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(4*hidden_channels, 4*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(4*hidden_channels, 2*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(2*hidden_channels, hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(hidden_channels, _differentiator_out_channels, aggr='lstm'),'x, edge_index -> x'),
        ])

        _integrator_in_channels = n_fields + n_fields
        _integrator_out_channels = n_fields
        self.integrator = gnn.Sequential( 'x, edge_index',[
            (SAGEConv(_integrator_in_channels, hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            # (gnn.LayerNorm(hidden_channels), 'x->x'),
            (self.act, 'x->x'),
            (SAGEConv(hidden_channels, 2*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(2*hidden_channels, 4*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(4*hidden_channels, 4*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(4*hidden_channels, 2*hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(2*hidden_channels, hidden_channels, aggr='lstm'),'x, edge_index -> x'),
            (self.act, 'x->x'),
            (SAGEConv(hidden_channels, _integrator_out_channels, aggr='lstm'),'x, edge_index -> x'),
        ])

        self.reset_parameters()
    
    def reset_parameters(self):
        self.differentiator.reset_parameters()
        self.integrator.reset_parameters()
    
    def forward(self, data, n_time=10, delta_t=0.02, device=torch.device('cpu')):
        ##
        F_0 = data.pressure.float()[:,0].unsqueeze(1)
        edge_index = data.edge_index
        meshfield = data.node_attr.float()
        ##
        if isinstance(meshfield, Tensor):
            meshfield = (meshfield, None)
        ##
        F_dots, Fs = [], []
        F_current = F_0
        for i in range(1, n_time + 1):
            x = torch.cat([F_current, meshfield[0]], dim=1)
            F_dot_current = self.differentiator(x, edge_index)
            if self.integrator is not None:
                x = torch.cat([F_current, F_dot_current], dim=1)
                F_next = F_current + self.integrator(x, edge_index)
            # else:
                # F_next = approximation_scheme(F_current, F_dot_current, delta_t, 'forward_euler')
            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)

# def approximation_scheme(x, dx, delta_t, scheme_type='forward_euler'):
#     if scheme_type == 'forward_euler':
#         return x + delta_t * dx
#     if scheme_type == 'crank_nicolson':
#         return x + delta_t * (dx[0] + dx[1]) / 2
