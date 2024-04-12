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

###################################################################################

class GraphUNetv2(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        block_depth: int = 2,
        dropout: float = None,
        *args, **kwargs
    ) -> None:
        ##
        super().__init__(*args, **kwargs)
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = pool_ratios
        self.sum_res = sum_res
        self.act = activation_resolver(act) 
        ##
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(GCNBlock(in_channels, hidden_channels, depth=block_depth, 
                                        act=self.act, norm=False, dropout=dropout))
        for i in range(depth):
            self.pools.append(gnn.TopKPooling(hidden_channels, self.pool_ratios))
            self.down_convs.append(GCNBlock(hidden_channels, hidden_channels, depth=block_depth,
                                        act=self.act, norm=False, dropout=dropout))
        
        in_channels = hidden_channels if sum_res else 2 * hidden_channels

        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNBlock(in_channels, hidden_channels, depth=block_depth,
                                        act=self.act, norm=False, dropout=dropout))
        self.up_convs.append(GCNBlock(in_channels, out_channels, depth=block_depth,
                                    act=self.act, norm=False, dropout=dropout))
        
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: OptTensor = None) -> Tensor:
        ##
        edge_weight = x.new_ones(edge_index.size(1))
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, 
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)
            
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
        
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x
        
        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
    
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
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mesh_decriptor = GraphUNetv2(
            in_channels=n_meshfield,
            hidden_channels=hidden_size,
            out_channels=latent_size,
            depth=4,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator = GraphUNetv2(
            in_channels=n_field+latent_size+use_time_feature,
            hidden_channels=hidden_size,
            out_channels=n_field,
            depth=4,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        # self.integrator = GraphUNetv2(
        #     in_channels=n_field*2,
        #     hidden_channels=hidden_size,
        #     out_channels=n_field,
        #     depth=2,
        #     pool_ratios=0.5,
        #     sum_res=False,
        #     act=act,
        #     dropout=dropout
        # )
        self.integrator = gnn.MLP(
            in_channels=n_field*2,
            hidden_channels=hidden_size,
            out_channels=n_field,
            num_layers=4,
            act=act
        )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        ##
        F_dots, Fs = [], []
        F_current = F_0
        for i in range(n_time):
            if time is None:
                x = torch.cat([F_current, meshfield], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                x = torch.cat([F_current, meshfield, _time_current], dim=1)
            F_dot_current = self.differentiator(x, edge_index)
            F_dot_current = F.tanh(F_dot_current)

            x = torch.cat([F_current, F_dot_current], dim=-1)
            F_next = self.integrator(x, edge_index)
            F_next = F.tanh(F_next)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)


###################################################################################

class RecurrentFormulationNet_reduced(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: int,
        hidden_size: int,
        latent_size: int,
        act='relu',
        use_time_feature: bool = False,
        dropout: float = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mesh_decriptor = GraphUNetv2(
            in_channels=n_meshfield,
            hidden_channels=hidden_size,
            out_channels=latent_size,
            depth=3,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator = GraphUNetv2(
            in_channels=n_field+latent_size+use_time_feature,
            hidden_channels=hidden_size,
            out_channels=n_field,
            depth=3,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        ##
        F_dots, Fs = [], []
        F_current = F_0
        dt = 4.0 / 200
        for i in range(n_time):
            if time is None:
                x = torch.cat([F_current, meshfield], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                x = torch.cat([F_current, meshfield, _time_current], dim=1)
            F_dot_current = self.differentiator(x, edge_index)
            F_dot_current = F.tanh(F_dot_current)

            # F_next = F_current + self.integrator(F_dot_current, edge_index)
            F_next = F_current + F_dot_current * dt
            # F_next = F.tanh(F_next)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)

###################################################################################

class RecurrentFormulationNet_hidden(nn.Module):
    def __init__(self, 
        n_field: int,
        n_meshfield: int,
        hidden_size: int,
        latent_size: int,
        act='relu',
        use_time_feature: bool = False,
        dropout: float = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        self.mesh_decriptor = GraphUNetv2(
            in_channels=n_meshfield,
            hidden_channels=hidden_size,
            out_channels=latent_size,
            depth=4,
            block_depth=2,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator1 = GraphUNetv2(
            in_channels=n_field+latent_size+use_time_feature+hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            depth=5,
            block_depth=2,
            pool_ratios=0.5,
            sum_res=False,
            act=act,
            dropout=dropout
        )

        self.differentiator2 = nn.Linear(hidden_size, n_field)

        self.integrator = gnn.MLP(
            in_channels=n_field*2,
            hidden_channels=hidden_size,
            out_channels=n_field,
            num_layers=2,
            act=act
        )
    
    def forward(self, F_0, edge_index, meshfield, time=None, n_time=1, device=None):
        ##
        meshfield = self.mesh_decriptor(meshfield, edge_index)
        meshfield = F.tanh(meshfield)
        ##
        F_dots, Fs = [], []
        F_current = F_0
        F_hidden = torch.zeros((F_0.size(0), self.hidden_size)).float().to(device)
        for i in range(n_time):
            if time is None:
                x = torch.cat([F_current, meshfield, F_hidden], dim=1)
            else:
                _time_current = time[:, i].unsqueeze(1)
                x = torch.cat([F_current, meshfield, _time_current, F_hidden], dim=1)
            F_hidden = self.differentiator1(x, edge_index)
            F_hidden = F.mish(F_hidden)
            F_dot_current = self.differentiator2(F_hidden)
            F_dot_current = F.tanh(F_dot_current)
            x = torch.cat([F_current, F_dot_current], dim=-1)
            F_next = self.integrator(x, edge_index)
            F_next = F.tanh(F_next)

            Fs.append(F_next.unsqueeze(1))
            F_dots.append(F_dot_current.unsqueeze(1))
            F_current = F_next.detach()
        
        return torch.cat(Fs, dim=1), torch.cat(F_dots, dim=1)