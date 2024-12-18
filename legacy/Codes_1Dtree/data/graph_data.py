import networkx as nx
from torch_geometric.data import Data
from typing import Optional
from torch import Tensor


class TorchGraphData(Data):
    r'''
    Graph data class expanded from torch_geometric.data.Data()
    Store a single graph data
    '''
    def __init__(self, 
        edge_index:Optional[Tensor] = None,
        edge_attr:Optional[Tensor] = None, 
        x:Optional[Tensor] = None, 
        y:Optional[Tensor] = None, 
        pos:Optional[Tensor] = None,  **kwargs
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
    
    @property
    def graph(self):
        edgelist = list(map(tuple, self.edge_index.numpy().transpose()))
        return nx.from_edgelist(edgelist=edgelist)
        
    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def number_of_edges(self):
        return self.graph.number_of_edges()

    def draw(self, **kwargs):
        nx.draw(self.graph, **kwargs)
    
    @property
    def number_of_timesteps(self):
        if self.pressure is not None:
            return self.pressure.size(1)
        elif self.flowrate is not None:
            return self.flowrate.size(1)
        else:
            return 0