import torch_geometric.nn as gnn
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor
import torch.nn.functional as F
import torch

def GCNBlock(in_channels, out_channels, batch_norm=True):
    _layers = [(gnn.GCNConv(in_channels, out_channels), 'x, edge_index -> x')]
    if batch_norm:
        _layers.append((gnn.BatchNorm(out_channels), 'x -> x'))
    return gnn.Sequential('x, edge_index', _layers)

class SurfaceRegresionNet(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        aggr: Union[str, List[str]] = 'sum',
        act=F.relu,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.act = act

        self.input = GCNBlock(in_channels, hidden_channels, batch_norm=True)

        self.down1 = GCNBlock(hidden_channels, 2*hidden_channels, batch_norm=True)
        self.down2 = GCNBlock(2*hidden_channels, 4*hidden_channels, batch_norm=True)
        self.down3 = GCNBlock(4*hidden_channels, 8*hidden_channels, batch_norm=True)

        self.latten = GCNBlock(8*hidden_channels, 8*hidden_channels, batch_norm=True)

        self.up3 = GCNBlock(16*hidden_channels, 4*hidden_channels, batch_norm=True)
        self.up2 = GCNBlock(8*hidden_channels, 2*hidden_channels, batch_norm=True)
        self.up1 = GCNBlock(4*hidden_channels, hidden_channels)

        self.output1 = GCNBlock(2*hidden_channels, hidden_channels, batch_norm=True)
        self.output2 = nn.Linear(hidden_channels, hidden_channels)
        self.output3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, 
        x: Tensor, 
        edge_index: Tensor
    ):
        x = self.act(self.input(x, edge_index))

        x1 = self.act(self.down1(x, edge_index))
        x2 = self.act(self.down2(x1, edge_index))
        x3 = self.act(self.down3(x2, edge_index))

        x4 = self.act(self.latten(x3, edge_index))

        x4 = torch.cat([x4, x3], dim=1)
        x5 = self.act(self.up3(x4, edge_index))
        x5 = torch.cat([x5, x2], dim=1)
        x6 = self.act(self.up2(x5, edge_index))
        x6 = torch.cat([x6, x1], dim=1)
        x7 = self.act(self.up1(x6, edge_index))

        x7 = torch.cat([x7, x], dim=1)
        output = self.act(self.output1(x7, edge_index))
        output = self.act(self.output2(output))
        output = self.output3(output)

        return output