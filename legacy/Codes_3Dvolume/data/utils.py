import numpy as np
import torch
from torch import Tensor
from typing import Optional, Callable, Union, List, Tuple, Dict

from data.graph_data import TorchGraphData


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

