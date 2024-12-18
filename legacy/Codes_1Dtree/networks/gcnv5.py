import torch
import torch_scatter
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.resolver import activation_resolver
import torch.nn.functional as F
from torch_geometric.typing import OptTensor
from typing import Tuple



# class ProcessorLayer(gnn.MessagePassing):
#     def __init__(self, in_channels, out_channels, act='relu', **kwargs):
#         super(ProcessorLayer, self).__init__(  **kwargs )
#         self.edge_mlp = nn.Sequential(nn.Linear( 3* in_channels , out_channels),
#                                    nn.ReLU(),
#                                    nn.Linear( out_channels, out_channels),
#                                    nn.LayerNorm(out_channels))

#         self.node_mlp = nn.Sequential(nn.Linear( 2* in_channels , out_channels),
#                                    nn.ReLU(),
#                                    nn.Linear( out_channels, out_channels),
#                                    nn.LayerNorm(out_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.edge_mlp[0].reset_parameters()
#         self.edge_mlp[2].reset_parameters()

#         self.node_mlp[0].reset_parameters()
#         self.node_mlp[2].reset_parameters()

#     def forward(self, x, edge_index, edge_attr, size = None):
#         out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]

#         updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation

#         updated_nodes = x + self.node_mlp(updated_nodes) # residual connection

#         return updated_nodes, updated_edges

#     def message(self, x_i, x_j, edge_attr):
#         updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
#         updated_edges=self.edge_mlp(updated_edges)+edge_attr

#         return updated_edges

#     def aggregate(self, updated_edges, edge_index, dim_size = None, reduce = 'sum'):
#         # The axis along which to index number of nodes.
#         node_dim = 0

#         out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = reduce)

#         return out, updated_edges

class Input(gnn.MessagePassing):
    def __init__(self, node_in_channels, out_channels, edge_in_channels=0, aggr='sum', **kwargs):
        super(Input, self).__init__(aggr,  **kwargs  )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_in_channels+edge_in_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        node, edge = self.propagate(
            edge_index=edge_index,
            node_attr=x,
            edge_attr=edge_attr
        )
        return node

    def message(self, node_attr_i, node_attr_j, edge_attr=None):
        edge = torch.cat([node_attr_i, node_attr_j], dim=1)
        if edge_attr is not None:
            edge = torch.cat([edge, edge_attr], dim=1)
        edge = self.edge_mlp(edge)
        return edge

    def aggregate(self, edge, edge_index, dim_size=None):
        node = torch_scatter.scatter(edge, edge_index[1, :], dim=0, reduce=self.aggr)
        return node, edge

class RecurrentFormulationNet(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: Tuple[ int, int ], # (node_size, edge_size)
        hidden_size: int,
        n_hidden: int,
        act='relu',
        dropout: float = None,
        use_hidden: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_hidden = use_hidden
        self.hidden_size = hidden_size
        self.act = activation_resolver(act)
        
        self.input = Input(n_field + n_meshfield[0] + use_hidden*hidden_size, hidden_size, n_meshfield[1])
        
        # self.net = gnn.GCN(in_channels=hidden_size, hidden_channels=hidden_size, num_layers=n_hidden, out_channels=hidden_size, \
        #                 dropout=dropout, act=self.act)
        self.net = gnn.GraphUNet(in_channels=hidden_size, hidden_channels=hidden_size,
                                out_channels=hidden_size, depth=5, pool_ratios=0.2, act=self.act)

        self.output = gnn.GCNConv(hidden_size*2, n_field)
    
    def forward(self, F_0, edge_index, meshfield, bc=None, n_time=1, device=None):
        ##
        Fs = []
        F_current = F_0
        if self.use_hidden:
            h = torch.zeros(size=F_0.size()[0:1]+(self.hidden_size,)).float().cuda()
        for _ in range(n_time):
            node_attr = torch.cat([F_current, meshfield[0]], dim=1)
            if self.use_hidden:
                # print(node_attr.size(), h.size())
                node_attr = torch.cat([node_attr, h], dim=1)
            edge_attr = meshfield[1]
            
            h1 = self.input(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
            h1 = self.act(h1)

            h2 = self.net(x=h1, edge_index=edge_index)
            h = h2.detach()
            h2 = self.act(h2)

            F_next = self.output(x=torch.cat([h1,h2], dim=1), edge_index=edge_index)
            # F_next = F.tanh(F_next)
            Fs.append(F_next.unsqueeze(1))
            
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1)
    

class RecurrentFormulationNetv2(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: Tuple[ int, int ], # (node_size, edge_size)
        hidden_size: int,
        n_hidden: int,
        act='relu',
        dropout: float = None,
        use_hidden: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_hidden = use_hidden
        self.hidden_size = hidden_size
        self.act = activation_resolver(act)
        
        # self.input = Input(n_field + n_meshfield[0] + use_hidden*hidden_size, hidden_size, n_meshfield[1])
        
        # # self.net = gnn.GCN(in_channels=hidden_size, hidden_channels=hidden_size, num_layers=n_hidden, out_channels=hidden_size, \
        # #                 dropout=dropout, act=self.act)
        # self.net = gnn.GraphUNet(in_channels=hidden_size, hidden_channels=hidden_size,
        #                         out_channels=hidden_size, depth=5, pool_ratios=0.2, act=self.act)

        # self.output = gnn.GCNConv(hidden_size*2, n_field)

        self.gcn_encoder = gnn.GCN(
            in_channels=hidden_size, 
            hidden_channels=hidden_size, 
            num_layers=n_hidden, 
            out_channels=hidden_size,
            dropout=dropout, 
            act=self.act
        )

        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=5)
    
    def forward(self, F_0, edge_index, meshfield, bc=None, n_time=1, device=None):
        ##
        Fs = []
        F_current = F_0
        if self.use_hidden:
            h = torch.zeros(size=F_0.size()[0:1]+(self.hidden_size,)).float().cuda()
        for _ in range(n_time):
            node_attr = torch.cat([F_current, meshfield[0]], dim=1)
            if self.use_hidden:
                # print(node_attr.size(), h.size())
                node_attr = torch.cat([node_attr, h], dim=1)
            edge_attr = meshfield[1]
            
            h1 = self.input(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
            h1 = self.act(h1)

            h2 = self.net(x=h1, edge_index=edge_index)
            h = h2.detach()
            h2 = self.act(h2)

            F_next = self.output(x=torch.cat([h1,h2], dim=1), edge_index=edge_index)
            # F_next = F.tanh(F_next)
            Fs.append(F_next.unsqueeze(1))
            
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1)