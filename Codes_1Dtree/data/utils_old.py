import os
import numpy as np
import torch
from torch import Tensor
# from torch_geometric.data import Dataset
from typing import Optional, Callable, Union, List, Tuple
from data.graph_data import TorchGraphData
from sklearn.preprocessing import PowerTransformer
from torch_geometric.utils import subgraph


##########################################################################################
def read_1D_input(
        file_name : str,
        var_dict = {
            'node_attr' : ['x_end', 'y_end', 'z_end'], 
            'edge_index' : ['ID', 'PareID'], 
            'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol0', 'Vol1', 'Vol1-0'],
        }
    ):
    r"""Read Output_subject_Amount_St_whole.dat
    Data stored in edge-wise format
    Data format
    ID PareID Length Diameter ... Vol1-0 Vol0 Vol1
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith branch----------)
    -  -      -      -        ... -      -    -
    """
    def _float(str):
        '''Change node type to float'''
        _dict = {'C':0, 'P':1, 'E':2, 'G':3, 'T':4}
        try:
            return float(str)
        except:
            return _dict[str]
    _vectorized_float = np.vectorize(_float)

    # Read file
    file = open(file_name, 'r')
    header = file.readline()
    data = file.read()
    file.close()

    # Process header, read number of variables
    vars = header.replace('\n',' ')
    vars = vars.split(' ')
    vars = list(filter(None, vars))
    n_var = len(vars)

    # Process data, split column and row
    data = data.replace('\n',' ')
    data = data.split(' ')
    data = list(filter(None, data))

    data = np.array(data).reshape((-1, n_var)).transpose()
    data_dict = {}
    for i in range(len(vars)):
        data_dict[vars[i]] = _vectorized_float(data[i])
    
    # Calculate volume fraction
    data_dict['Vol0-fraction'] = data_dict['Vol0'] / np.sum(data_dict['Vol0'])
    data_dict['Vol1-fraction'] = data_dict['Vol1'] / np.sum(data_dict['Vol1'])
    data_dict['Vol1-0-fraction'] = data_dict['Vol1-0'] / np.sum(data_dict['Vol1-0'])
    
    # Store output variables
    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
        for data_var in var_dict[var]:
            out_dict[var].append(data_dict[data_var])
        if len(out_dict[var]) == 1:
            out_dict[var] = out_dict[var][0]

    # Reshape and update output variables
    
    out_dict['edge_index'] = np.array(out_dict['edge_index'], dtype=np.int32)

    out_dict['node_attr'] = edge_to_node(np.array(out_dict['node_attr'],\
                            dtype=np.float32).transpose(), out_dict['edge_index'])
    
    out_dict['edge_attr'] = np.array(out_dict['edge_attr'], dtype=np.float32).transpose()
    
    out_dict['node_attr'][0,0] = data_dict['x_start'][0] #
    out_dict['node_attr'][0,1] = data_dict['y_start'][0] # Insert entrance node
    out_dict['node_attr'][0,2] = data_dict['z_start'][0] #

    return out_dict


##########################################################################################
def read_1D_output(
        file_names,
        var_dict = {
            'pressure' : 'p',
            'flowrate' : 'flowrate'
        }
    ):
    r"""Read data_plt_nd/plt_nd_000time.dat (all time_id)
    Data stored in node wise format
    Data format
    VARIABLES="x" "y" "z" "p" ... "flowrate"  "resist" "area"                                    
     ZONE T= "plt_nd_000time.dat                                 "
     N=       xxxxx , E=       xxxxx ,F=FEPoint,ET=LINESEG
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith node----------)
    -  -      -      -        ... -      -    -
    -  -
    -  -
    (---------connectivity of jth branch-------)
    -  -
    """
    # Read variable list and n_node, n_edge
    file = open(file_names[0], 'r')
    line = file.readline()
    line = line.replace('VARIABLES',' ')
    line = line.replace('=',' ')
    line = line.replace('\n',' ')
    line = line.replace('"',' ')
    vars = list(filter(None, line.split(' ')))
    n_var = len(vars)

    file.readline()
    line = file.readline()
    line = line.split(',')
    n_node = int(line[0].replace('N=',' ').replace(' ',''))
    n_edge = int(line[1].replace('E=',' ').replace(' ',''))
    file.close()

    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
    # Read all time id
    for file_name in file_names:
        # Skip header and read data part
        file = open(file_name,'r')
        file.readline()
        file.readline()
        file.readline()
        data = file.read()
        file.close()

        # Process data string into numpy array of shape=(n_node, n_var)
        data = data.replace('\n',' ')
        data = list(filter(None, data.split(' ')))
        edge_index = data[n_var*n_node:n_var*n_node + 2 * n_edge]
        data = np.array(data[0:n_var*n_node], dtype=np.float32)
        data = data.reshape((n_node, n_var)).transpose()
        
        # Store to variable dict
        for var in var_dict:
            out_dict[var].append(np.expand_dims(data[vars.index(var_dict[var])], axis=-1))
        
    # Aggregate results from all time id.
    for var in var_dict:
        out_dict[var] = np.concatenate(out_dict[var], axis=-1)
    edge_index = np.array(edge_index, dtype = np.int32).reshape((n_edge, 2)).transpose() - 1
    return out_dict


##########################################################################################
def node_to_edge(node_attr, edge_index):
    r'''Change node wise features to edge wise features'''
    return np.array([node_attr[i] for i in edge_index[1]])


##########################################################################################
def edge_to_node(edge_attr, edge_index):
    r'''Change edge wise features to node wise features'''
    n_node = edge_index.max() + 1
    if len(edge_attr.shape) <=1:
        n_attr = 1
        node_attr = np.zeros(shape=(n_node,) , dtype=np.float32)
    else:
        n_attr = edge_attr.shape[1]
        node_attr = np.zeros(shape=(n_node, n_attr) , dtype=np.float32)
    for i in range(edge_index.shape[1]):
        node_attr[edge_index[1][i]] = edge_attr[i]
    # find root and assign root features
    is_child = np.isin(edge_index[0], edge_index[1])
    root = np.where(is_child == False)[0][0]
    node_attr[edge_index[0][root]] = edge_attr[root]
    return node_attr


##########################################################################################
def read_sh_file_v1(
    file_name: str='./pres_flow_lung.sh',
    var_dict = {
        'global_attr': ['v_TLC', 'v_FRC', 'v_tidal', 'gender', 'age', 'height', 'acoef', 'bcoef'],
        'total_time': ['tperiod'],
        'rho': ['rhog'],
        'vis': ['visg'],
    }
):
    r'''read pres_flow_lung.sh file'''
    # read file as string and process
    _file = open(file_name, 'r+')
    file_str = _file.read()
    lines = file_str.split('\n')

    file_dict = {}
    for i in range(len(lines)):
        lines[i] = lines[i].split('#')[0]
        lines[i] = lines[i].replace(' ','')
            
        line = lines[i].split('=')
        if len(line) == 2:
            file_dict[line[0]] = line[1]

    out_dict = {}
    
    # Store output variables
    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
        for data_var in var_dict[var]:
            out_dict[var].append(float(file_dict[data_var].replace('d','E')))
        if len(out_dict[var]) == 1:
            out_dict[var] = out_dict[var][0]
        else:
            out_dict[var] = np.array(out_dict[var])
    return out_dict


##########################################################################################
def cal_deriv_F(F, step, dim=1):
    r'''calculate time derivative of input fields'''
    if isinstance(F, torch.Tensor):
        F = F.numpy()
    if dim >= len(np.shape(F)):
        return None
    _F = np.swapaxes(F, 0, dim)

    _F_dot = []
    for i in range(0, _F.shape[0]-1):
        _F_dot.append(np.expand_dims((_F[i] - _F[i-1])/step, 0))
    _F_dot.append(np.expand_dims((_F[-1] - _F[-2])/step, 0))

    _F_dot = np.concatenate(_F_dot, axis=0)
    _F_dot = np.swapaxes(_F_dot, dim, 0)
    return _F_dot


##########################################################################################
def cut_branch(data: TorchGraphData, max_gen: int):
    r'''cut sub branches to the given generation'''
    
    gen = data.edge_attr.numpy()[:,2]
    remain_edge = np.where(gen <= max_gen)[0] # branch id after cutting
    remain_node = data.edge_index.numpy()[:,remain_edge]
    remain_node = np.unique(np.concatenate([remain_node[0], remain_node[1]], axis=0))
    
    return get_graph_partition(data, remain_node, recursive=False)


##########################################################################################
def get_graph_partition(
    data : TorchGraphData,
    partition : np.array,
    recursive : bool,
    list_of_node_features = ['node_attr', 'pressure', 'flowrate', 'flowrate_bc', \
                            'is_terminal', 'time', 'pressure_dot', 'flowrate_dot'],
    list_of_edge_features = ['edge_attr']
) -> TorchGraphData:
    '''Return sub-graph data for given list of node index in a parition'''
    edge_index = data.edge_index.numpy()

    # Mark all edges containing nodes in partition
    edge_mark = np.isin(edge_index, partition)
    if recursive:
        edge_mark = np.logical_or(edge_mark[0], edge_mark[1])
    else:
        edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
    # Get global id of all edges containing nodes in partition
    partition_edge_id = np.argwhere(edge_mark == True).squeeze(1)
    # Get edge_index of partition (with global node id)
    partition_edge_index = edge_index[:, partition_edge_id]
    # Get global id of all nodes in parition (for recursive only)
    partition_node_id = np.unique(np.concatenate(list(partition_edge_index) + [partition]))

    #### Convert global node id to partition node id
    index = lambda n : list(partition_node_id).index(n)
    v_index = np.vectorize(index)
    # Convert global node id to partition node id in partition_edge_index
    if partition_edge_index.shape[1] > 0:
        partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
    #### Get partition of all features
    partition_data = TorchGraphData()
    for key in data._store:
        if key in list_of_node_features:
            setattr(partition_data, key, data._store[key][partition_node_id])
        elif key in list_of_edge_features:
            setattr(partition_data, key, data._store[key][partition_edge_id])
        elif key == 'edge_index':
            setattr(partition_data, key, partition_edge_index)
        else:
            setattr(partition_data, key, data._store[key])
    return partition_data


# def get_graph_partition_v1(
#     data : TorchGraphData,
#     partition : np.array,
#     recursive : bool,
#     list_of_node_features = ['node_attr', 'pressure', 'flowrate', 'flowrate_bc', \
#                             'is_terminal', 'time', 'pressure_dot', 'flowrate_dot'],
#     list_of_edge_features = ['edge_attr']
# ) -> TorchGraphData:
#     '''Return sub-graph data for given list of node index in a parition'''
#     edge_index = data.edge_index.numpy()

#     # Mark all edges containing nodes in partition
#     edge_mark = np.isin(edge_index, partition)
#     if recursive:
#         edge_mark = np.logical_or(edge_mark[0], edge_mark[1])
#     else:
#         edge_mark = np.logical_and(edge_mark[0], edge_mark[1])
#     # Get global id of all edges containing nodes in partition
#     partition_edge_id = np.argwhere(edge_mark == True).squeeze(1)
#     # Get edge_index of partition (with global node id)
#     partition_edge_index = edge_index[:, partition_edge_id]
#     # Get global id of all nodes in parition (for recursive only)
#     partition_node_id = np.unique(np.concatenate(list(partition_edge_index) + [partition]))

#     #### Convert global node id to partition node id
#     index = lambda n : list(partition_node_id).index(n)
#     v_index = np.vectorize(index)
#     # Convert global node id to partition node id in partition_edge_index
#     if partition_edge_index.shape[1] > 0:
#         partition_edge_index = torch.tensor(v_index(partition_edge_index))
    
#     #### Get partition of all features
#     partition_data = TorchGraphData()
#     for key in data._store:
#         if key in list_of_node_features:
#             setattr(partition_data, key, data._store[key][partition_node_id])
#         elif key in list_of_edge_features:
#             setattr(partition_data, key, data._store[key][partition_edge_id])
#         elif key == 'edge_index':
#             setattr(partition_data, key, partition_edge_index)
#         else:
#             setattr(partition_data, key, data._store[key])
#     return partition_data