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

from neuralop.models.fno import FNO
