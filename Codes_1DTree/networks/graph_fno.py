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
from torch_geometric.nn import MessagePassing, GraphSAGE
from neuralop.models.fno import FNO

class GraphFNO(nn.Module):
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

        if isinstance(n_meshfields, int):
            n_meshfields = (n_meshfields, 0)

        self.operator = FNO(
            n_modes=(16, 32), 
            in_channels = n_meshfields[0],
            out_channels = n_fields,
            n_layers = num_layers
        )
    
    def forward(self, data, n_time=1, delta_t=0.02):
        pass