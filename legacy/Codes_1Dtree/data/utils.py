import numpy as np
import torch
from torch import Tensor
from typing import Optional, Callable, Union, List, Tuple, Dict

from data.graph_data import TorchGraphData


###########################################################################

def to_float(s: str):
    _branch_type = {'C':0, 'P':1, 'E':2, 'G':3, 'T':4}
    try:
        return float(s)
    except:
        return _branch_type[s]
    
to_float_vectorized = np.vectorize(to_float)

###########################################################################
    
def read_1D_input(
        file_name: str,
        # variable_dict: Dict = {
        #     'edge_index' : ['PareID', 'ID'],
        #     'node_attr' : ['x_end', 'y_end', 'z_end'],
        #     'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol0', 'Vol1', 'Vol1-0']
        # }
        variable_dict: Dict = {
            'edge_index' : ['PareID', 'ID'],
            'node_attr' : ['x_end', 'y_end', 'z_end', 'Length', 'Diameter',\
                        'Gene', 'Lobe']
        }
) -> Dict:
    ##
    _file = open(file_name, 'r+')
    _header_str = _file.readline()
    _data_str = _file.read()
    _file.close()

    ##
    _variables = list(filter(None ,_header_str.replace('\n',' ').split(' ')))
    number_of_variables = len(_variables)

    ##
    _data = list(filter(None, _data_str.replace('\n',' ').split(' ')))
    _data = np.array(_data).reshape((-1, number_of_variables))
    _data_dict = {}
    for i in range(number_of_variables):
        _data_dict[_variables[i]] = to_float_vectorized(_data[:,i]) 
    
    ##
    _output_dict = {}
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = []
        for _field in variable_dict[_output_variable]:
            _output_dict[_output_variable].append(_data_dict[_field])
        if len(_output_dict[_output_variable]) == 1:
            _output_dict[_output_variable] = _output_dict[_output_variable][0]
        _output_dict[_output_variable] = np.array(_output_dict[_output_variable])
    
    for _output_variable in variable_dict:
        if _output_variable == 'edge_index':
            continue
        _output_dict[_output_variable] = np.transpose(_output_dict[_output_variable])
    
    _output_dict['node_attr'] = edge_to_node(_output_dict['node_attr'], _output_dict['edge_index'])
    _output_dict['node_attr'][0,0] = _data_dict['x_start'][0] #
    _output_dict['node_attr'][0,1] = _data_dict['y_start'][0] # Insert entrance node
    _output_dict['node_attr'][0,2] = _data_dict['z_start'][0] #

    return _output_dict

###########################################################################

def read_1D_output(
        file_names,
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
    number_of_edges = int(_line[1].replace('E=',' ').replace(' ',''))
    _file.close()

    ##
    _output_dict = {}
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = []
    for file_name in file_names:
        ###
        _file = open(file_name)
        for _ in range(3):
            _file.readline()
        _data_str = _file.read()
        _file.close()
        ###
        _data = list(filter(None,_data_str.replace('\n',' ').split(' ')))
        # edge_index is already read in input file
        # _edge_index = _data[number_of_variables*number_of_nodes:
        #                    number_of_variables*number_of_nodes+2*number_of_edges]
        _data = np.array(_data[0:number_of_variables*number_of_nodes], dtype=np.float32) \
                .reshape((number_of_nodes, number_of_variables)).transpose()
        ###
        for _output_variable in variable_dict:
            _output_dict[_output_variable].append(
                np.expand_dims(_data[_variables.index(variable_dict[_output_variable])], axis=-1))

    ##
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = np.concatenate(_output_dict[_output_variable], axis=-1) 
    
    return _output_dict       

###########################################################################

def read_shell_script(
        file_name: str='./pres_flow_lung.sh',
        variable_dict: Dict={
            'global_attr': ['v_TLC', 'v_FRC', 'v_tidal', 'gender', 'age', 'height'],
            'total_time': ['tperiod'],
            'rho': ['rhog'],
            'vis': ['visg'],
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

###########################################################################

def calculate_derivative(
        input: Union[np.ndarray, Tensor], 
        h: Union[float, List[float], np.ndarray], 
        axis: int = 1
) -> Union[np.ndarray, Tensor]:
    if isinstance(input, np.ndarray):
        ##
        assert axis < len(input.shape)
        ##
        _input = np.swapaxes(input, 0, axis)
        _h = h
        ##
        _padding_prefix = _input[0] - (_input[1] - _input[0])
        _padding_suffix = _input[-1] - (_input[-2] - _input[-1])
        _input = np.insert(_input, 0, _padding_prefix)
        _input = np.append(_input, _padding_suffix)
        if isinstance(_h, List) or isinstance(_h, np.ndarray):
            _h = np.insert(_h, 0, _h[0])
            _h = np.append(_h, _h[-1])
        else:
            _h = np.full((_input.shape[0]-1,), _h)
        ##
        _derivative = []
        for i in range(1, _input.shape[0]-1):
            _derivative_i = (_input[i+1] - _input[i-1]) / (_h[i-1] + _h[i])
            _derivative.append(_derivative_i)
        _derivative = np.array(_derivative)
        _derivative = np.swapaxes(_derivative, axis, 0)
        return _derivative
    
    if isinstance(input, Tensor):
        ##
        assert axis < len(input.size())
        ##
        _input = torch.transpose(input, 0, axis)
        _h = h
        ##
        _padding_prefix = _input[0] - (_input[1] - _input[0])
        _padding_suffix = _input[-1] - (_input[-2] - _input[-1])
        _input = torch.cat([_padding_prefix.unsqueeze(0), _input, _padding_suffix.unsqueeze(0)], axis=0)
        if isinstance(_h, List) or isinstance(_h, np.ndarray):
            _h = np.insert(_h, 0, _h[0])
            _h = np.append(_h, _h[-1])
        else:
            _h = np.full((_input.shape[0]-1,), _h)
        ##
        _derivative = []
        for i in range(1, _input.size(0)-1):
            _derivative_i = (_input[i+1] - _input[i-1]) / (_h[i-1] + _h[i])
            _derivative.append(_derivative_i.unsqueeze(0))
        _derivative = torch.cat(_derivative, axis=0)
        _derivative = torch.transpose(_derivative, axis, 0)
        return _derivative

###########################################################################

from torch_geometric.utils import k_hop_subgraph
def get_subgraph(
        data: TorchGraphData,
        subset: Union[int, List[int], Tensor],
        number_of_hops: int = 1,
        list_of_node_features: List = ['node_attr', 'pressure', 'flowrate', 'flowrate_bc', \
                                'is_terminal', 'time', 'pressure_dot', 'flowrate_dot'],
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

###########################################################################

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

###########################################################################

def edge_to_node(
        edge_attr: np.ndarray,
        edge_index: np.ndarray
) -> np.ndarray:
    ##
    edge_index = edge_index.astype(int)
    number_of_nodes = edge_index.max() + 1
    number_of_attrs = edge_attr.shape[1]
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

    return _node_attr

    # node_ID = np.sort(np.unique(np.concatenate([edge_index[0], edge_index[1]])))
    # node_attr = []

    # for i in node_ID:
    #     temp = np.isin(edge_index[1], i)
    #     # temp = np.logical_or(temp[0], temp[1])
    #     temp = np.where(temp == True)[0]
    #     temp = edge_attr[temp]
    #     # temp = np.mean(temp, axis=0)
    #     temp = temp[0]
    #     temp = np.expand_dims(temp, 0)
    #     node_attr.append(temp)

    # node_attr = np.concatenate(node_attr, axis=0)
    # return node_attr
        
    