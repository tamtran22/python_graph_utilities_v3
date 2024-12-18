import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from typing import Optional, Callable, Union, List, Tuple
from data.graph_data import TorchGraphData
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from data.utils import *
from torch_geometric.loader import DataLoader


########################################################################

class OneDDatasetBuilder(Dataset):
    r'''
    Graph data class expanded from torch_geometric.data.Dataset()
    Build and store multiple graph datas
    '''
    def __init__(self,
        raw_dir: Optional[str] = None, # Path to raw data files
        root_dir: Optional[str] = None, # Path to store processed data files
        sub_dir: Optional[str] = 'processed',
        subjects: Optional[Union[str, List[str]]] = 'all',
        time_names: Optional[Union[str, List[str]]] = 'all',
        data_type = torch.float32,
        readme=None
    ):
        ##
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.data_type = data_type
        self.readme = readme
        self._get_subject_names(subjects, raw_dir)
        self._get_time_names(time_names)
        super().__init__(root_dir, transform, pre_transform, pre_filter)

    def _get_subject_names(self, subjects, subject_dir):
        ##
        if (subjects == 'all') or (subjects is None):
            _file_names = os.listdir(subject_dir)
            self.subjects = _file_names
        else:
            self.subjects = subjects

    def _get_time_names(self, time_name):
        self.time_names = time_name

    def processed_file_names(self):
        return [f'{self.root}/{self.sub}/{subject}.pt' for subject in self.subjects]

    def len(self):
        return len(self.subjects)

    def __getitem__(self, index):
        return torch.load(self.processed_file_names()[index])

    def get(self, index):
            return torch.load(self.processed_file_names()[index])
    
    def process(self):
        ##
        CFD_1D_dir = 'CFD_1D'
        file_name_input = lambda subject : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/Output_{subject}_Amount_St_whole.dat'
        file_name_output = lambda subject, time : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/data_plt_nd/plt_nd_000{time}.dat'
        file_name_sh = lambda subject : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/pres_flow_lung.sh'
        ##
        for subject in self.subjects:
            ###
            print(f'Process subject number {self.subjects.index(subject)}, subject name : {subject}.')
            ###
            _input_file_name = file_name_input(subject)
            _data_dict_input = read_1D_input(_input_file_name)
            _output_file_names = [file_name_output(subject, time) for time in self.time_names]
            _data_dict_output = read_1D_output(_output_file_names)
            _shellscript_file_name = file_name_sh(subject)
            _data_dict_sh = read_shell_script(_shellscript_file_name)
            # ###
            # _data = TorchGraphData()
            # for _variable in _data_dict_input:
            #     setattr(_data, _variable, torch.tensor(_data_dict_input[_variable]).type(self.data_type))
            # for _variable in _data_dict_output:
            #     setattr(_data, _variable, torch.tensor(_data_dict_output[_variable]).type(self.data_type))
            # for _variable in _data_dict_sh:
            #     setattr(_data, _variable, torch.tensor(_data_dict_sh[_variable]).type(self.data_type))
            # _data.edge_index = _data.edge_index.type(torch.LongTensor)
            ###
            _node_attr = torch.tensor(_data_dict_input['node_attr']).type(self.data_type)
            # _edge_attr = torch.tensor(_data_dict_input['edge_attr']).type(self.data_type)
            _edge_index = torch.tensor(_data_dict_input['edge_index']).type(torch.LongTensor)
            _pressure = torch.tensor(_data_dict_output['pressure']).type(self.data_type)
            _flowrate = torch.tensor(_data_dict_output['flowrate']).type(self.data_type)
            _global_attr = torch.tensor(_data_dict_sh['global_attr']).type(self.data_type)

            _bc = _flowrate[0,:]
            _total_time = _data_dict_sh['total_time']
            # _number_of_nodes = _pressure.size(0)
            _number_of_timesteps = _pressure.size(1)
            # _h = _total_time / (_number_of_timesteps - 1)
            _global_attr = torch.cat([_global_attr.unsqueeze(0)]*_node_attr.size(0), dim=0)
            _node_attr = torch.cat([_node_attr, _global_attr], dim=1)
            # _pressure_dot = calculate_derivative(_pressure, h=_h, axis=1)
            # _flowrate_dot = calculate_derivative(_flowrate, h=_h, axis=1)
            # _time = torch.cat([torch.arange(0, _total_time + _h, step=_h).unsqueeze(0)]*_number_of_nodes, dim=0)

            _data = TorchGraphData()
            setattr(_data, 'node_attr', _node_attr)
            # setattr(_data, 'edge_attr', _edge_attr)
            setattr(_data, 'edge_index', _edge_index)
            setattr(_data, 'pressure', _pressure)
            setattr(_data, 'flowrate', _flowrate)
            setattr(_data, 'boundary_condition', _bc)
            # setattr(_data, 'pressure_dot', _pressure_dot)
            # setattr(_data, 'flowrate_dot', _flowrate_dot)
            # setattr(_data, 'time', _time)
            ###   
            torch.save(_data, self.processed_file_names()[self.subjects.index(subject)])
        ##
        if self.readme is not None:
            _file_readme = open(f'{self.root}/readme.txt', 'w+')
            _file_readme.write(self.readme)
            _file_readme.close()

########################################################################

class OneDDatasetLoader(Dataset):
    r'''
    Graph data class expanded from torch_geometric.data.Dataset()
    Load multiple graph datas
    '''
    def __init__(self,
        raw_dir: Optional[str] = None, # Path to raw data files
        root_dir: Optional[str] = None, # Path to store processed data files
        sub_dir: Optional[str] = 'processed',
        subjects: Optional[Union[str, List[str]]] = 'all',
        time_names: Optional[Union[str, List[str]]] = 'all',
        data_type = torch.float32
    ):
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.data_type = data_type
        self._get_subject_names(subjects, f'{root_dir}/{sub_dir}')
        self._get_time_names(time_names)
        super().__init__(root_dir, transform, pre_transform, pre_filter)

    def _get_subject_names(self, subjects, subject_dir):
        ##
        if (subjects == 'all') or (subjects is None):
            _file_names = os.listdir(subject_dir)
            _filename_filter = lambda s : not s in [
                'pre_filter.pt', 'pre_transform.pt', 'batched_id.pt', 'batched_info.pt'
            ]
            _file_names = list(filter(_filename_filter, _file_names))
            _subjects = [subject.replace('.pt','',subject.count('.pt')) for subject in _file_names]
            self.subjects = _subjects
        else:
            self.subjects = subjects

    def _get_time_names(self, time_names):
        self.time_names = time_names

    def processed_file_names(self):
        return [f'{self.root}/{self.sub}/{subject}.pt' for subject in self.subjects]

    def len(self):
        return len(self.subjects)

    def __getitem__(self, index):
        return torch.load(self.processed_file_names()[index])

    def get(self, index):
            return torch.load(self.processed_file_names()[index])
    
    def process(self):
        pass

    def load_scaler(self, variable, index=None):
        if not os.path.isdir(f'{self.root}/scaler'):
            return None
        if index is None:
            _scaler_file = open(f'{self.root}/scaler/{variable}.pkl','rb')
        else:
            _scaler_file = open(f'{self.root}/scaler/{variable}_{index}.pkl','rb')
        _scaler = pickle.load(_scaler_file)
        _scaler_file.close()
        return _scaler

################################################################################
def resolve_scaler(scaler_type, **kwargs):
    if scaler_type == 'minmax_scaler':
        return MinMaxScaler(**kwargs)
    if scaler_type == 'standard_scaler':
        return StandardScaler()
    if scaler_type == 'robust_scaler':
        return RobustScaler()
################################################################################
def normalize(
    dataset: Union[OneDDatasetBuilder, OneDDatasetLoader],
    sub_dir: str = 'normalized',
    scaler_dict: Dict = {
        'node_attr': ('minmax_scaler', 0, None), # (scaler_type, scale_dim, clipping_range)
        'output': ('minmax_scaler', 0, 0.99) # (min, max) or quantile_coeff
    }
) -> OneDDatasetLoader:
    ##
    if sub_dir == '' or sub_dir == '/':
        print('Unable to clear root folder!')
    else:
        os.system(f'rm -rf {dataset.root}/{sub_dir}')
    os.system(f'mkdir {dataset.root}/{sub_dir}')
    if not os.path.isdir(f'{dataset.root}/scaler'):
        os.system(f'mkdir {dataset.root}/scaler')
    if dataset.len() <= 0:
        return dataset
    ##
    for _variable in scaler_dict:
        ### 
        # check if variable is normalized per subject or whole dataset
        if 'per_subject' in scaler_dict[_variable][0]:
            continue
        ###
        _variable_data = []
        for i in range(dataset.len()):
            _variable_data.append(dataset[i]._store[_variable])
        _variable_data = torch.cat(_variable_data, dim=0)
        if scaler_dict[_variable][1] is None:
            _variable_data = _variable_data.flatten().unsqueeze(1)
        elif len(_variable_data.shape) <= 1:
            _variable_data = _variable_data.unsqueeze(1)
        if scaler_dict[_variable][2] is not None:
            if isinstance(scaler_dict[_variable][2], float):
                _min = np.quantile(_variable_data.numpy(), q=1.-scaler_dict[_variable][2])
                _max = np.quantile(_variable_data.numpy(), q=scaler_dict[_variable][2])
            else:
                (_min, _max) = scaler_dict[_variable][2]
            _variable_data = torch.clip(_variable_data, _min, _max)
        ###
        _scaler_type = scaler_dict[_variable][0].replace('_per_subject','')
        _scaler = resolve_scaler(_scaler_type, feature_range=(-1.,1.))
        _scaler.fit(_variable_data)
        _scaler_file = open(f'{dataset.root}/scaler/{_variable}.pkl','wb')
        pickle.dump(_scaler, _scaler_file)
        _scaler_file.close()
    ##
    for i in range(dataset.len()):
        _data = dataset[i]
        _normalized_data = TorchGraphData()
        for _variable in _data._store:
            if _variable in scaler_dict:
                _variable_data = _data._store[_variable]
                _resize = _variable_data.size()
                if scaler_dict[_variable][1] is None:
                    _variable_data = _variable_data.flatten()
                if len(_variable_data.size()) <=1:
                    _variable_data = _variable_data.unsqueeze(1)
                if scaler_dict[_variable][2] is not None:
                    if isinstance(scaler_dict[_variable][2], float):
                        _min = np.quantile(_variable_data.numpy(), q=1.-scaler_dict[_variable][2])
                        _max = np.quantile(_variable_data.numpy(), q=scaler_dict[_variable][2])
                    else:
                        (_min, _max) = scaler_dict[_variable][2]
                    _variable_data = torch.clip(_variable_data, _min, _max)
                if 'per_subject' in scaler_dict[_variable][0]:
                    _scaler_type = scaler_dict[_variable][0].replace('_per_subject','')
                    _scaler = resolve_scaler(_scaler_type, feature_range=(-1.,1.))
                    _scaler.fit(_variable_data)
                    _scaler_file = open(f'{dataset.root}/scaler/{_variable}_{i}.pkl','wb')
                    pickle.dump(_scaler, _scaler_file)
                    _scaler_file.close()
                else:
                    _scaler = dataset.load_scaler(_variable)
                _variable_data = torch.tensor(_scaler.transform(_variable_data))
                _variable_data = torch.reshape(_variable_data, _resize)
            else:
                _variable_data = _data._store[_variable]
            setattr(_normalized_data, _variable, _variable_data)
        _file_name = dataset.processed_file_names()[i].replace(f'{dataset.root}','').replace(f'{dataset.sub}','').replace('/','')
        torch.save(_normalized_data, f'{dataset.root}/{sub_dir}/{_file_name}')
    
    return OneDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)

##########################################################################

def batchgraph_generation_wise(
        dataset: Union[OneDDatasetBuilder, OneDDatasetLoader],
        sub_dir: str = 'batched',
        batch_gens: List[Tuple] = [[0,9,1],[10,13,1], [14, 15,1], [16, 18,1], [18,50,1]],
        timestep: int = None,
        timeslice_hops: int = 1,
        timeslice_steps=1
) -> OneDDatasetLoader:
    ##
    if sub_dir == '' or sub_dir == '/':
        print('Unable to clear root folder!')
    else:
        os.system(f'rm -rf {dataset.root}/{sub_dir}')
    os.system(f'mkdir {dataset.root}/{sub_dir}')
    if dataset.len() <= 0:
        return dataset
    ##
    _batched_dataset = []
    _batched_dataset_id = []
    for i in range(dataset.len()):
        ###
        if dataset.sub == 'processed':
            _gen = dataset[i].node_attr[:,5].type(torch.LongTensor)
        else:
            _scaler = dataset.load_scaler('node_attr')
            _node_attr = _scaler.inverse_transform(dataset[i].node_attr)
            _gen = _node_attr[:,5].astype(np.int32)
        # _batch_gens = [list(range(batch_gens[i][0], batch_gens[i][1])) for i in range(len(batch_gens))]
        _batch_gens = []
        for batch_gen in batch_gens:
            _batch_gen = [list(range(batch_gens[i][0], batch_gens[i][1]))]
            _batch_gen *= batch_gen[2]
            _batch_gens += _batch_gen
        _subsets = [list(np.where(np.isin(_gen, _batch_gens[i]))[0]) for i in range(len(batch_gens))]
        ###
        _subgraphs = get_batchgraphs(
            data=dataset[i],
            subsets=_subsets,
            subset_size=1,
            subset_hops=1,
            timestep=timestep,
            timeslice_hops=timeslice_hops,
            timeslice_steps=timeslice_steps
        )
        _batched_dataset += _subgraphs
        _batched_dataset_id += [i]*len(_subgraphs)
    ##
    for i in range(len(_batched_dataset)):
        torch.save(_batched_dataset[i], f'{dataset.root}/{sub_dir}/subgraph_{i}.pt')
    torch.save(torch.tensor(_batched_dataset_id), f'{dataset.root}/{sub_dir}/batched_id.pt')
    return OneDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)

#############################################################################

def batchgraph(
        dataset: Union[OneDDatasetBuilder, OneDDatasetLoader],
        sub_dir: str = 'batched',
        batchsize: int = None,
        timestep: int = None,
        timeslice_hops: int = 1,
        timeslice_steps=1
) -> OneDDatasetLoader:
    ##
    if sub_dir == '' or sub_dir == '/':
        print('Unable to clear root folder!')
    else:
        os.system(f'rm -rf {dataset.root}/{sub_dir}')
    os.system(f'mkdir {dataset.root}/{sub_dir}')
    if dataset.len() <= 0:
        return dataset
    ##
    _batched_dataset = []
    _batched_dataset_id = []
    for i in range(dataset.len()):
        ###

        _subgraphs = get_batchgraphs(
            data=dataset[i],
            subsets=None,
            subset_size=batchsize,
            subset_hops=2,
            timestep=timestep,
            timeslice_hops=timeslice_hops,
            timeslice_steps=timeslice_steps
        )
        _batched_dataset += _subgraphs
        _batched_dataset_id += [i]*len(_subgraphs)
    ##
    for i in range(len(_batched_dataset)):
        torch.save(_batched_dataset[i], f'{dataset.root}/{sub_dir}/subgraph_{i}.pt')
    torch.save(torch.tensor(_batched_dataset_id), f'{dataset.root}/{sub_dir}/batched_id.pt')
    return OneDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)

##############################################################################

def dataset_to_loader(
    dataset : Dataset,
    data_subset_dict = {
        'train': [],
        'test': [],
        'validation': []
    },
    n_data_per_batch = 1
):
    # Get batching id
    if 'batched' in dataset.sub:
        _batching_id = torch.load(f'{dataset.root}/{dataset.sub}/batched_id.pt').numpy()
        print('batching_id', _batching_id)
        for _data_subset in data_subset_dict:
            print(data_subset_dict[_data_subset])
            print(np.isin(_batching_id, data_subset_dict[_data_subset] ))
            data_subset_dict[_data_subset] = list(np.where(np.isin(_batching_id, data_subset_dict[_data_subset] ) == True)[0])
            print(data_subset_dict[_data_subset])
    _data_subsets = []
    for _data_subset in data_subset_dict:
        _data_subsets.append(DataLoader([dataset[i] for i in data_subset_dict[_data_subset]], batch_size=n_data_per_batch))
    return tuple(_data_subsets)

##########################################################################
from torch_geometric.loader import NeighborLoader
def batchgraph_neighbor(
        dataset: Union[OneDDatasetBuilder, OneDDatasetLoader],
        sub_dir: str = 'batched',
        batch_size: int = 100,
) -> OneDDatasetLoader:
    ##
    if sub_dir == '' or sub_dir == '/':
        print('Unable to clear root folder!')
    else:
        os.system(f'rm -rf {dataset.root}/{sub_dir}')
    os.system(f'mkdir {dataset.root}/{sub_dir}')
    if dataset.len() <= 0:
        return dataset
    ##
    _batched_dataset = []
    _batched_dataset_id = []
    for i in range(dataset.len()):
        data_loader = NeighborLoader(dataset[i], num_neighbors=[1], batch_size=batch_size)
        _subgraphs = [data for (_, data) in list(enumerate(data_loader))]
        _batched_dataset += _subgraphs
        _batched_dataset_id += [i]*len(_subgraphs)
    ##
    for i in range(len(_batched_dataset)):
        torch.save(_batched_dataset[i], f'{dataset.root}/{sub_dir}/subgraph_{i}.pt')
    torch.save(torch.tensor(_batched_dataset_id), f'{dataset.root}/{sub_dir}/batched_id.pt')
    return OneDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)