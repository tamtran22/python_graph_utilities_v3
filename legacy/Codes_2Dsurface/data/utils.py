import numpy as np
import math
import torch
from torch import Tensor
from typing import Optional, Callable, Union, List, Tuple, Dict

from data.graph_data import TorchGraphData


###########################################################################

def to_number(s: str):
    try:
        if s.isdigit():
            return int(s)
        return float(s)
    except:
        return s
    
# to_float_vectorized = np.vectorize(to_float)

###########################################################################
import re
def read_2Dvolume_data(
        file_name,
        variable_dict: Dict = {
            'node_attr' : ['x', 'y', 'z'],
            'output' : ['wss']
        }
) -> List[Dict]:
    ##
    _file = open(file_name, 'r+')
    _file.readline()
    _line = _file.readline().replace('variables',' ').replace('=',' ').replace('\n',' ').replace('"',' ').replace(',','')
    _variables = list(filter(None, _line.split(' ')))
    _number_of_variables = len(_variables)
    _zones = _file.read().split('zone')[1:]
    _file.close()
    
    ##
    _output_dicts = []
    for _zone in _zones:
        ###
        _str = re.findall(r'\s\w+=\w+', _zone)
        _params = {}
        for _s in _str:
            __s = _s.replace(' ','').split('=')
            _params[__s[0]] = to_number(__s[1])
        _number_of_nodes = _params['Nodes']
        _number_of_elements = _params['Elements']
        _zone_type = _params['ZoneType']
        _data_packing_type = _params['DataPacking']

        ### Data packing type is Block
        if _data_packing_type == 'Block':
            _output_dict = {}
            ###
            _variable_location = re.findall(r'\[[0-9,]+\]=CellCentered', _zone)
            _cell_centered_variables = []
            if len(_variable_location) > 0:
                _s = _variable_location[0].replace('CellCentered','').replace('=','').replace(' ','').replace('[','').replace(']','')
                _cell_centered_variables += [to_number(i) - 1 for i in _s.split(',')]
            ###
            _str = re.findall(r'[\s-]\d.\d+E.\d+', _zone)
            _data = np.array(_str, dtype=np.float32)
            _data_dict = {}
            _pointer = 0
            for i in range(_number_of_variables):
                if i in _cell_centered_variables:
                    _number_of_variables_data = _number_of_elements
                else:
                    _number_of_variables_data = _number_of_nodes
                _data_dict[_variables[i]] = _data[_pointer:_pointer+_number_of_variables_data]
                _pointer += _number_of_variables_data
            ###
            _str = re.findall(r'\s+\d+\s+\d+\s+\d+', _zone)
            _elements = []
            for _s in _str:
                _element = np.array(list(filter(None, _s.replace('\n','').split(' '))), dtype=np.int32)
                _elements.append(_element - 1)
            _elements = np.array(_elements, dtype=np.int32)
            
            ### processed data fields
            ###
            _edge_index = element_to_graph(_elements, type='CellCentered')
            ###
            _node_xyz = np.array([_data_dict['x'], _data_dict['y'], _data_dict['z']], dtype=np.float32).transpose()
            _elements_node_xyz = []
            for _element in _elements:
                _element_node_xyz = np.array([], dtype=np.float32)
                for _node in _element:
                    _element_node_xyz = np.append(_element_node_xyz, _node_xyz[_node])
                _elements_node_xyz.append(np.array(_element_node_xyz))
            _elements_node_xyz = np.array(_elements_node_xyz, dtype=np.float32)
            ###
            _wss = np.array(_data_dict['wss'], dtype=np.float32)
            ###
            _elements_area = np.expand_dims(elements_area(_elements, _node_xyz), axis=1)
            
            ###
            _output_dict['node_xyz'] = _node_xyz
            _output_dict['elements'] = _elements
            _output_dict['edge_index'] = _edge_index
            _output_dict['node_attr'] = np.concatenate((_elements_node_xyz, _elements_area), axis=1)
            _output_dict['output'] = np.clip(_wss, 0, 0.1)
            ###
            _output_dicts.append(_output_dict)
    
    return _output_dicts   

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

def bidirect_graph(
        edge_index: Union[np.ndarray, Tensor]
) -> Union[np.ndarray, Tensor]:
    ##
    if isinstance(edge_index, np.ndarray):
        if (len(edge_index.shape) == 2) and (edge_index.shape[0] == 2):
            _reversed_edge_index = np.array([edge_index[1],edge_index[0]])
            return np.concatenate([edge_index, _reversed_edge_index], axis=1)
        return np.array(0)
    ##
    if isinstance(edge_index, Tensor):
        if (len(edge_index.size()) == 2) and (edge_index.size(0) == 2):
            _reversed_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
            return torch.cat([edge_index, _reversed_edge_index], dim=1)
        return torch.tensor(0)
    ##
    return np.array(0)

##########################################################################

def unique_edge_graph(
        edge_index: np.ndarray,
        edge_attr: Optional[Union[np.ndarray, List[np.ndarray]]] = None
):
    ##
    _edge_index, _index = np.unique(edge_index, axis=1, return_index=True)
    if edge_attr is None:
        return _edge_index
    elif isinstance(edge_attr, List):
        _edge_attr = [attr[_index] for attr in edge_attr]
        return _edge_index, edge_attr
    elif isinstance(edge_attr, np.ndarray):
        _edge_attr = edge_attr[_index]
        return _edge_index, _edge_attr


###########################################################################
# working
def write_zone_to_tec(
        file_name: str,
        zone_datas: List[TorchGraphData],
) -> None:
    ##
    _file = open(file_name, 'w+')
    _file.write('Title="pred.dat"\n')
    _file.write('Variables="x","y","z","output"\n')
    ##
    for i in range(len(zone_datas)):
        data = zone_datas[i]
        _file.write(f'Zone T="zone_{i}"\n')
        _file.write(f'Nodes={data["node_xyz"].shape[0]}, Elements={data["elements"].shape[0]}, ZoneType=FETriangle, DataPacking=Block, VarLocation=([4]=CellCentered)\n')
        _x = data["node_xyz"][:,0]
        _y = data["node_xyz"][:,1]
        _z = data["node_xyz"][:,2]
        _output = data["output"]
        _elements = data["elements"] + 1
        ###
        for i in range(len(_x)):
            _file.write('  ')
            _file.write("{:.6E}".format(_x[i]))
            if (i%5==4)and(i<len(_x)-1):
                _file.write('\n')
        _file.write('\n')
        ###
        for i in range(len(_y)):
            _file.write('  ')
            _file.write("{:.6E}".format(_y[i]))
            if (i%5==4)and(i<len(_y)-1):
                _file.write('\n')
        _file.write('\n')
        ###
        for i in range(len(_z)):
            _file.write('  ')
            _file.write("{:.6E}".format(_z[i]))
            if (i%5==4)and(i<len(_z)-1):
                _file.write('\n')
        _file.write('\n')
        ###
        for i in range(len(_output)):
            _file.write('  ')
            _file.write("{:.6E}".format(_output[i]))
            if (i%5==4)and(i<len(_output)-1):
                _file.write('\n')
        _file.write('\n')
        ###
        for i in range(len(_elements)):
            for j in range(len(_elements[i])):
                _file.write(f'  {_elements[i][j]}')
            _file.write('\n')
    ##
    _file.close()

###########################################################################

def element_to_graph(
        elements: Union[np.ndarray, List],
        type: str = 'CellCentered' #'CellVertexs'
) -> np.ndarray:
    ##
    if isinstance(elements, List):
        elements = np.array(elements, dtype=np.int32)
    ##
    number_of_elements = elements.shape[0]
    number_of_nodes_per_element = elements.shape[1]
    ##
    if type == 'CellCentered':
        _edge_index = np.array([[],[]], dtype=np.int32)
        for i in range(number_of_elements):
            _neighbor_elements = np.isin(elements, elements[i])
            _neighbor_elements = np.logical_or.reduce(_neighbor_elements, axis=1)
            _neighbor_elements = np.where(_neighbor_elements == True)[0]
            for j in _neighbor_elements:
                _intersection = list(set(elements[i]).intersection(set(elements[j])))
                if len(_intersection) == 2:
                    _edge_index = np.append(_edge_index, [[i],[j]], axis=1)
        return _edge_index
    ##
    if type == 'CellVertexs':
        _edge_index = np.array([[],[]], dtype=np.int32)
        for i in range(number_of_elements):
            e = elements[i]
            _edge_index = np.append(_edge_index,[
                [e[0],e[0],e[1],e[1],e[2],e[2]],
                [e[1],e[2],e[2],e[0],e[0],e[1]]
            ], axis=1)
        _edge_index = np.unique(_edge_index, axis=1)
        return _edge_index

###########################################################################

def _triangle_area(points: Union[np.ndarray, List]):
    a = points[0]
    b = points[1]
    c = points[2]
    ab = math.sqrt(np.sum(np.square(a-b)))
    bc = math.sqrt(np.sum(np.square(b-c)))
    ca = math.sqrt(np.sum(np.square(a-c)))
    s = (ab+bc+ca) / 2
    return math.sqrt(s*(s-ab)*(s-bc)*(s-ca))

###########################################################################

def elements_area(
        elements: Union[np.ndarray, List],
        nodes_xyz: Union[np.ndarray, List]
) -> np.ndarray:
    ##
    if isinstance(elements, List):
        elements = np.array(elements, dtype=np.int32)
    if isinstance(nodes_xyz, List):
        nodes_xyz = np.array(nodes_xyz, dtype=np.int32)
    _element_area = np.array([], dtype=np.float32)
    ##
    number_of_elements = elements.shape[0]
    number_of_nodes_per_element = elements.shape[1]
    if number_of_nodes_per_element == 3:
        area = _triangle_area
    ##
    _element_areas = []
    for i in range(number_of_elements):
        _element_areas.append(area([nodes_xyz[j] for j in elements[i]]))
    
    return np.array(_element_areas, dtype=np.float32)

