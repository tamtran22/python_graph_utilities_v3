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

# ###########################################################################
    
# def read_1D_input(
#         file_name: str,
#         variable_dict: Dict = {
#             'edge_index' : ['PareID', 'ID'],
#             'node_attr' : ['x_end', 'y_end', 'z_end'],
#             'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol0', 'Vol1', 'Vol1-0']
#         }
# ) -> Dict:
#     ##
#     _file = open(file_name, 'r+')
#     _header_str = _file.readline()
#     _data_str = _file.read()
#     _file.close()

#     ##
#     _variables = list(filter(None ,_header_str.replace('\n',' ').split(' ')))
#     number_of_variables = len(_variables)

#     ##
#     _data = list(filter(None, _data_str.replace('\n',' ').split(' ')))
#     _data = np.array(_data).reshape((-1, number_of_variables))
#     _data_dict = {}
#     for i in range(number_of_variables):
#         _data_dict[_variables[i]] = to_float_vectorized(_data[:,i])
    
#     ##
#     _output_dict = {}
#     for _output_variable in variable_dict:
#         _output_dict[_output_variable] = []
#         for _field in variable_dict[_output_variable]:
#             _output_dict[_output_variable].append(_data_dict[_field])
#         if len(_output_dict[_output_variable]) == 1:
#             _output_dict[_output_variable] = _output_dict[_output_variable][0]
#         _output_dict[_output_variable] = np.array(_output_dict[_output_variable])

#     return _output_dict

###########################################################################

def read_3Dvolume_data(
        file_names,
        variable_dict: Dict = {
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'u': 'u',
            'v': 'v',
            'w': 'w',
            'p': 'p',
            'f': 'f',
            'vmag': 'vmag'
        }
) -> Dict:
    ##
    _file = open(file_names[0], 'r+')
    _line = _file.readline().replace('variables',' ').replace('=',' ').replace('\n',' ').replace('"',' ').replace(',','')
    _variables = list(filter(None, _line.split(' ')))
    number_of_variables = len(_variables)
    _line = _file.readline().replace('\n','').replace(' ','').replace('zoneT=','').replace('f','').replace('_','').replace('"','')
    _time = float(_line)
    _line = _file.readline().split(',')
    number_of_nodes = int(_line[0].replace('N=',' ').replace(' ',''))
    number_of_edges = int(_line[1].replace('E=',' ').replace(' ',''))
    _file.close()

    ##
    _output_dict = {}
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = []
    _element = None
    for file_name in file_names:
        ###
        _file = open(file_name)
        for _ in range(3):
            _file.readline()
        _data_str = _file.read()
        _file.close()
        ###
        _data = list(filter(None,_data_str.replace('\n',' ').split(' ')))
        if _element is None:
            _element = np.array(_data[number_of_variables*number_of_nodes:
                    number_of_variables*number_of_nodes+4*number_of_edges], dtype=np.int32) \
                    .reshape((number_of_edges, 4))
        _data = np.array(_data[0:number_of_variables*number_of_nodes], dtype=np.float32) \
                .reshape((number_of_nodes, number_of_variables)).transpose()
        ###
        for _output_variable in variable_dict:
            _output_dict[_output_variable].append(
                np.expand_dims(_data[_variables.index(variable_dict[_output_variable])], axis=-1))

    ##
    for _output_variable in variable_dict:
        _output_dict[_output_variable] = np.concatenate(_output_dict[_output_variable], axis=-1) 
    ##
    _edge_index = []
    for e in _element:
        _edge_index.append(np.array([[e[0],e[1]]]))
        _edge_index.append(np.array([[e[0],e[2]]]))
        _edge_index.append(np.array([[e[0],e[3]]]))
        _edge_index.append(np.array([[e[1],e[2]]]))
        _edge_index.append(np.array([[e[1],e[3]]]))
        _edge_index.append(np.array([[e[2],e[3]]]))

        _edge_index.append(np.array([[e[1],e[0]]]))
        _edge_index.append(np.array([[e[2],e[0]]]))
        _edge_index.append(np.array([[e[3],e[0]]]))
        _edge_index.append(np.array([[e[2],e[1]]]))
        _edge_index.append(np.array([[e[3],e[1]]]))
        _edge_index.append(np.array([[e[3],e[2]]]))
    _output_dict['edge_index'] = np.unique(np.concatenate(_edge_index, axis=0), axis=0)
    return _output_dict       

###########################################################################

def read_shell_script(
        file_name: str='./pres_flow_lung.sh',
        variable_dict: Dict={
            'global_attr': ['v_TLC', 'v_FRC', 'v_tidal', 'gender', 'age', 'height', 'acoef', 'bcoef'],
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
        input: np.array, 
        h: Union[float, List[float], np.ndarray], 
        axis: int = 1
) -> np.array:
    ##
    if axis >= len(input.shape):
        return None # Differentiate axis is different from input axis
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
        print(_h[i-1], _h[i])
        _derivative_i = (_input[i+1] - _input[i-1]) / (_h[i-1] + _h[i])
        _derivative.append(_derivative_i)
    _derivative = np.array(_derivative)
    _derivative = np.swapaxes(_derivative, axis, 0)
    return _derivative

###########################################################################

from torch_geometric.utils import k_hop_subgraph
def subgraph(
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

