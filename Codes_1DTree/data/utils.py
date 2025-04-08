import numpy as np
import torch
from typing import Dict, Tuple, List, Union
from data.graph_data import TorchGraphData
from torch import Tensor

############################################################
def read_1D_input(
        file_name: str
) -> Dict:
    ##
    file = open(file_name, 'r+')
    header = file.readline()
    data = file.read()
    file.close()

    ##
    variable = list(filter(None, header.replace('\n', ' ').split(' ')))
    num_var = len(variable)

    ##
    data = list(filter(None, data.replace('\n',' ').split(' ')))
    data = np.array(data).reshape((-1, num_var))
    output = {}
    for i in range(num_var):
        output[variable[i]] = data[:,i]
    return output


#############################################################
def read_1D_output(
        file_names: List[str],
        variable_dict: Dict = {
            'pressure' : 'p',
            'flowrate' : 'flowrate'
        }
) -> Dict:
    ##
    _file = open(file_names[0], 'r+')
    _line = _file.readline().replace('VARIABLES',' ').replace('=',' ').replace('\n',' ').replace('"',' ')
    _variables = list(filter(None, _line.split(' ')))
    number_of_variables = len(_variables)
    _file.readline()
    _line = _file.readline().split(',')
    number_of_nodes = int(_line[0].replace('N=',' ').replace(' ',''))
    # number_of_edges = int(_line[1].replace('E=',' ').replace(' ',''))
    _file.close()

    ##
    _output_dict = {}
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = []
    # edge_index = None
    for file_name in file_names:
        ###
        _file = open(file_name)
        for _ in range(3):
            _file.readline()
        _data_str = _file.read()
        _file.close()
        ###
        _data = list(filter(None,_data_str.replace('\n',' ').split(' ')))
        ###
        # if edge_index is None:                    
        #     edge_index = _data[number_of_variables*number_of_nodes:
        #                     number_of_variables*number_of_nodes+2*number_of_edges]
        _data = np.array(_data[0:number_of_variables*number_of_nodes]) \
                .reshape((number_of_nodes, number_of_variables)).transpose()
        ###
        for _output_variable in variable_dict:
            _output_dict[_output_variable].append(
                np.expand_dims(_data[_variables.index(variable_dict[_output_variable])], axis=-1))

    ##
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = np.concatenate(_output_dict[_output_variable], axis=-1) 
    
    return _output_dict

#####################################################################
def read_shell_script(
        file_name: str='./pres_flow_lung.sh',
        variable_dict: Dict={
            'global_attr': ['v_TLC', 'v_FRC', 'v_tidal', 'gender', 'age', 'height']
        }
) -> Dict:
    ##
    _file = open(file_name, 'r+')
    _file_str = _file.read()
    _file.close()
    
    ##
    _line = _file_str.split('\n')
    _file_dict = {}
    for i in range(len(_line)):
        _line[i] = _line[i].split('#')[0].replace(' ','').split('=')
        if len(_line[i]) == 2:
            _file_dict[_line[i][0]] = _line[i][1]
    
    ##
    _output_dict = {}
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = []
        for _field in variable_dict[_output_variable]:
            _output_dict[_output_variable].append(float(_file_dict[_field].replace('d', 'E')))
        if len(_output_dict[_output_variable]) == 1:
            _output_dict[_output_variable] = _output_dict[_output_variable][0]
        else:
            _output_dict[_output_variable] = np.array(_output_dict[_output_variable])
    
    return _output_dict

#############################################################
def process_data(
        data: Dict,
        attr_list: List = ['coordinate', 'length', 'diameter', 'generation', 
                           'lobe', 'flag', 'pressure', 'flowrate'],
        max_length: float = 0,
) -> Tuple:
    ## Edge_index
    parent_id = data['PareID'].astype(np.int32)
    child_id = data['ID'].astype(np.int32)
    edge_index = np.concatenate([
        np.expand_dims(parent_id, 0),
        np.expand_dims(child_id, 0)
    ], axis=0)

    ##
    coordinate = None
    if 'coordinate' in attr_list:
        coordinate = np.concatenate([
            np.expand_dims(data['x_end'].astype(np.float64), axis=1),
            np.expand_dims(data['y_end'].astype(np.float64), axis=1),
            np.expand_dims(data['z_end'].astype(np.float64), axis=1)
        ], axis=1)
        coordinate, root_node = edge_to_node(coordinate, edge_index)
        coordinate[root_node] = np.array([data['x_start'][0], data['y_start'][0], data['z_start'][0]])
    ##
    length = None
    if 'length' in attr_list:
        length = data['Length'].astype(np.float64)
    ##
    diameter = None
    if 'diameter' in attr_list:
        diameter = data['Diameter'].astype(np.float64)
    ##
    generation = None
    if 'generation' in attr_list:
        generation = data['Gene'].astype(np.float64)
    ##
    lobe = None
    if 'lobe' in attr_list:
        lobe = data['Lobe'].astype(np.int32)
        lobe = process_lobe(lobe)
    ##
    flag = None
    if 'flag' in attr_list:
        flag = data['Flag']
        flag = process_flag(flag)
    ##
    pressure = None
    if 'pressure' in attr_list:
        pressure = data['pressure'].astype(np.float64)
    ##
    flowrate = None
    if 'flowrate' in attr_list:
        flowrate = data['flowrate'].astype(np.float64)
    
    ##
    node_attr = {
        'pressure' : pressure,
        'flowrate' : flowrate
    }

    edge_attr = {
        'length' : length,
        'diameter' : diameter,
        'generation' : generation,
        'lobe' : lobe,
        'flag' : flag
    }

    ##
    edge_index_raw = None
    if max_length > 0:
        edge_index_raw = edge_index
        edge_index, coordinate, node_attr, edge_attr, original_flag = refine_1Dmesh(
            edge_index=edge_index,
            coordinate=coordinate,
            node_attr=node_attr,
            edge_attr=edge_attr,
            max_length=max_length
        )
    
    ## 
    output = {}
    output['edge_index'] = edge_index
    output['edge_index_raw'] = edge_index_raw
    output['coordinate'] = coordinate
    output['original_flag'] = original_flag
    for attr in node_attr:
        output[attr] = node_attr[attr]
    for attr in edge_attr:
        output[attr] = edge_attr[attr]

    return output

####################################################################
def refine_1Dmesh(
        edge_index: np.ndarray, 
        coordinate: np.ndarray,
        node_attr: Dict = {
            'pressure' : None,
            'flowrate' : None
        },
        edge_attr: Dict = {
            'length' : None,
            'diameter' : None,
            'generation' : None,
            'lobe' : None,
            'flag' : None
        },
        max_length: float = 1.
):  
    deleted_branch = []
    num_branch = edge_index.shape[1]
    original_flag = [1]*coordinate.shape[0]
    for i in range(num_branch):
        (proximal, distal) = (edge_index[0,i], edge_index[1,i])
        
        d = distance(coordinate[proximal], coordinate[distal])
        if d > max_length:
            num_p = int(d / max_length)
            N = coordinate.shape[0]
            delta = (coordinate[distal] - coordinate[proximal]) / (num_p + 1)
            abs_delta = d / (num_p + 1)

            # Add branch
            concat_edge_index = np.concatenate([[[proximal], [N]]] + \
                                [[[N+i], [N+i+1]] for i in range(num_p-1)] + \
                                [[[N+num_p-1], [distal]]], axis=1)
            edge_index = np.concatenate([edge_index, concat_edge_index], axis=1)

            # Add node
            coordinate = np.concatenate([
                coordinate,
                [coordinate[distal] - j*delta for j in reversed(range(1, num_p+1))]
            ], axis=0)

            # Add optional features
            for attr in node_attr:
                node_attr[attr] = np.concatenate([
                    node_attr[attr],
                    [node_attr[attr][distal]] * (num_p)
                ], axis=0)

            for attr in edge_attr:
                if attr in ['flag']:
                    # assign_value = np.array([0, 0, 0, 0, 0, 1])
                    assign_value = 5
                elif attr in ['length']:
                    assign_value = abs_delta
                else:
                    assign_value = edge_attr[attr][i]
                edge_attr[attr] = np.concatenate([
                    edge_attr[attr],
                    [assign_value] * (num_p + 1)
                ], axis=0)

            deleted_branch.append(i)
            original_flag += [0]*num_p
    edge_index = np.delete(edge_index, deleted_branch, axis=1)
    original_flag = np.array(original_flag, dtype=np.int32)

    for attr in edge_attr:
        edge_attr[attr] = np.delete(edge_attr[attr], deleted_branch, axis=0)

    return edge_index, coordinate, node_attr, edge_attr, original_flag

#######################################################
def edge_to_node(
        edge_attr: np.ndarray,
        edge_index: np.ndarray
) -> np.ndarray:
    ##
    edge_index = edge_index.astype(int)
    number_of_nodes = edge_index.max() + 1
    number_of_attrs = edge_attr.shape[1] if len(edge_attr.shape) > 1 else 1
    number_of_edges = edge_attr.shape[0]
    if len(edge_attr.shape) <= 1:
        _node_attr = np.zeros(shape=(number_of_nodes,), dtype=np.float64)
    else:
        _node_attr = np.zeros(shape=(number_of_nodes, number_of_attrs), dtype=np.float64)
    for i in range(number_of_edges):
        _node_attr[edge_index[1][i]] = edge_attr[i]
    ##
    _edge_has_parrent = np.isin(edge_index[0], edge_index[1])
    _root_edge = np.where(_edge_has_parrent == False)[0][0]
    _node_attr[edge_index[0][_root_edge]] = edge_attr[_root_edge]

    return _node_attr, edge_index[0][_root_edge]

#######################################################
def process_lobe(lobe: np.ndarray):
    return lobe
    # encode_lobe = {
    #     0 : np.array([[1, 0, 0, 0, 0, 0]]),
    #     1 : np.array([[0, 1, 0, 0, 0, 0]]),
    #     2 : np.array([[0, 0, 1, 0, 0, 0]]),
    #     3 : np.array([[0, 0, 0, 1, 0, 0]]),
    #     4 : np.array([[0, 0, 0, 0, 1, 0]]),
    #     5 : np.array([[0, 0, 0, 0, 0, 1]]),
    # }
    # output = [encode_lobe[lobe[i]] for i in range(lobe.shape[0])]
    # return np.concatenate(output, axis=0)

#######################################################
def process_flag(flag: np.ndarray):
    # encode_flag = {
    #     'C' : np.array([[1, 0, 0, 0, 0, 0]]),
    #     'P' : np.array([[0, 1, 0, 0, 0, 0]]),
    #     'E' : np.array([[0, 0, 1, 0, 0, 0]]),
    #     'G' : np.array([[0, 0, 0, 1, 0, 0]]),
    #     'T' : np.array([[0, 0, 0, 0, 1, 0]]),
    # }
    encode_flag = {
        'C' : 0,
        'P' : 1,
        'E' : 2,
        'G' : 3,
        'T' : 4,
    }
    output = [encode_flag[flag[i]] for i in range(flag.shape[0])]
    # return np.concatenate(output, axis=0)
    return np.array(output, dtype=np.int32)

######################################################
def distance(A, B):
    return np.sqrt(np.sum(np.square(A - B)))

###########################################################################

from torch_geometric.utils import k_hop_subgraph
def get_subgraph(
        data: TorchGraphData,
        subset: Union[int, List[int], Tensor],
        number_of_hops: int = 1,
        list_of_node_features: List = ['node_attr', 'pressure', 'flowrate', 'flowrate_bc', \
                                'original_flag', 'time', 'pressure_dot', 'flowrate_dot'],
        list_of_edge_features: List = ['edge_attr']
) -> TorchGraphData:
    ##
    _subset_nodeid, _subset_edgeindex, _, _subset_edgemark = k_hop_subgraph(
        node_idx=subset,
        num_hops=number_of_hops,
        edge_index=data.edge_index
    )
    _subset_edgeid = torch.where(_subset_edgemark == True)[0]

    ##
    _subset_index = lambda i : (_subset_nodeid==i).nonzero().squeeze().item()
    _subset_index_vectorize = np.vectorize(_subset_index)
    _subset_edgeindex = torch.tensor(_subset_index_vectorize(_subset_edgeindex)).type(torch.LongTensor)
    
    ##
    _subset_data = TorchGraphData()
    for _variable in data._store:
        if _variable in list_of_node_features:
            setattr(_subset_data, _variable, data._store[_variable][_subset_nodeid])
        elif _variable in list_of_edge_features:
            setattr(_subset_data, _variable, data._store[_variable][_subset_edgeid])
        elif _variable == 'edge_index':
            setattr(_subset_data, _variable, _subset_edgeindex)
        else:
            setattr(_subset_data, _variable, data._store[_variable])

    return _subset_data

###########################################################################

def get_timeslice(
        data: TorchGraphData,
        timeslice: Union[int, List[int], np.ndarray],
        number_of_hops: int = 1,
        hop_step: int = 1,
        list_of_time_features: List = ['pressure', 'flowrate', 'flowrate_bc', 'pressure_dot', \
                                'flowrate_dot', 'time']
) -> TorchGraphData:
    ##
    if isinstance(timeslice, int):
        timeslice = [timeslice]
    if isinstance(timeslice, np.ndarray):
        timeslice = list(timeslice)
    for _ in range(number_of_hops):
        timeslice.insert(0, timeslice[0] - hop_step)
        timeslice.append(timeslice[-1] + hop_step)
    while timeslice[0] < 0:
        timeslice.pop(0)
    while timeslice[-1] >= data.number_of_timesteps:
        timeslice.pop(-1)
    ##
    _timeslice_data = TorchGraphData()
    for _variable in data._store:
        if _variable in list_of_time_features:
            setattr(_timeslice_data, _variable, data._store[_variable][:,timeslice])
        else:
            setattr(_timeslice_data, _variable, data._store[_variable])

    return _timeslice_data

##########################################################################

# import nxmetis
def get_batchgraphs(
        data: TorchGraphData,
        subsets: Union[List, np.ndarray] = None,
        subset_size: int = None, # (batch size, number of_hops)
        subset_hops: int = None,
        timestep: int = None, # (number of timesteps, number of hops, hop step)
        timeslice_hops: int = None,
        timeslice_steps: int = None,
) -> List[TorchGraphData]:
    ##
    _temp_subgraphs = []
    if (subset_size is not None):
        ###
        # if subsets is None:
        #     (_, subsets) = nxmetis.partition(G=data.graph, nparts=int(data.number_of_nodes / subset_size))
        for _subset in subsets:
            _temp_subgraphs.append(get_subgraph(data, _subset, subset_hops))
    else:
        _temp_subgraphs.append(data)
    
    ##
    if timestep is not None:
        _timeslices = []
        i = 0
        while i < data.number_of_timesteps - 1:
            i_start = i
            i_end = i + timestep
            _timeslice = np.arange(start=i_start, stop=i_end, step=timeslice_steps)
            _timeslice = _timeslice[(_timeslice >= 0) & (_timeslice < data.number_of_timesteps)]
            _timeslices.append(_timeslice)
            i = i_end
        _subgraphs = []
        for _subgraph in _temp_subgraphs:
            for _timeslice in _timeslices:
                _subgraphs.append(get_timeslice(_subgraph, _timeslice, timeslice_hops, timeslice_steps))
    else:
        _subgraphs = _temp_subgraphs  
    return _subgraphs

    