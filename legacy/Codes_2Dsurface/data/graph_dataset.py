import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from typing import Optional, Callable, Union, List, Tuple
from data.graph_data import TorchGraphData
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from data.utils import *


########################################################################

class TwoDDatasetBuilder(Dataset):
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
        data_type = torch.float32
    ):
        ##
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.data_type = data_type
        self._get_subject_names(subjects, raw_dir)
        self._get_time_names(time_names)

        super().__init__(root_dir, transform, pre_transform, pre_filter)

    def _get_subject_names(self, subjects, subject_dir):
        ##
        if (subjects == 'all') or (subjects is None):
            _file_names = os.listdir(subject_dir)
            self.subjects = [_file_name.replace('.dat','') for _file_name in _file_names]
        else:
            self.subjects = subjects

    def _get_time_names(self, time_name):
        self.time_names = time_name

    def processed_file_names(self):
        if not os.path.isdir(f'{self.root}/{self.sub}'):
            return []
        _filename_filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', \
                                                'batched_id.pt', 'batched_info.pt']
        _file_names = os.listdir(f'{self.root}/{self.sub}/')
        _file_names = list(filter(_filename_filter, _file_names))
        return [f'{self.root}/{self.sub}/{_file_name}' for _file_name in _file_names]

    def len(self):
        return len(self.processed_file_names())

    def __getitem__(self, index):
        return torch.load(self.processed_file_names()[index])

    def get(self, index):
            return torch.load(self.processed_file_names()[index])
    
    def process(self):
        ##
        file_name_dat = lambda subject : f'{self.raw}/{subject}.dat'
        for subject in self.subjects:
            ###
            print(f'Process subject number {self.subjects.index(subject)}, subject name : {subject}.')
            ###
            _file_name = file_name_dat(subject)
            _data_dict_zones = read_2Dvolume_data(_file_name)
            ###
            for i in range(len(_data_dict_zones)):
                _data_dict_zone = _data_dict_zones[i]
                _data = TorchGraphData()
                for _variable in _data_dict_zone:
                    setattr(_data, _variable, torch.tensor(_data_dict_zone[_variable]).type(self.data_type))
                _data.edge_index = _data.edge_index.type(torch.LongTensor)
                _data.elements = _data.elements.type(torch.LongTensor)

                torch.save(_data, f'{self.root}/{self.sub}/{subject}_{i}.pt')

########################################################################
                
import re
class TwoDDatasetLoader(Dataset):
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
            _filename_filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', 'batched_id.pt', 'batched_info.pt']
            _file_names = list(filter(_filename_filter, _file_names))
            _subjects = list(set([re.sub(r'_\d+.pt', '', _file_name) for _file_name in _file_names]))
            self.subjects = _subjects
        else:
            self.subjects = subjects

    def _get_time_names(self, time_names):
        self.time_names = time_names

    def processed_file_names(self):
        if not os.path.isdir(f'{self.root}/{self.sub}'):
            return []
        _filename_filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', \
                                                'batched_id.pt', 'batched_info.pt']
        _file_names = os.listdir(f'{self.root}/{self.sub}/')
        _file_names = list(filter(_filename_filter, _file_names))
        return [f'{self.root}/{self.sub}/{_file_name}' for _file_name in _file_names]

    def len(self):
        return len(self.processed_file_names())

    def __getitem__(self, index):
        return torch.load(self.processed_file_names()[index])

    def get(self, index):
            return torch.load(self.processed_file_names()[index])
    
    def process(self):
        pass

    def export_subject(self, subject) -> List[TorchGraphData]:
        ##
        if not subject in self.subjects:
            return []
        ##
        _export_file_names = []
        for _file_name in self.processed_file_names():
            if subject in _file_name:
                _export_file_names.append(_file_name)
        _export_file_names.sort()

        return [torch.load(_file_name) for _file_name in _export_file_names]
    
    def load_scaler(self, variable):
        if not os.path.isdir(f'{self.root}/scaler'):
            return None
        _scaler_file = open(f'{self.root}/scaler/{variable}.pkl','rb')
        _scaler = pickle.load(_scaler_file)
        _scaler_file.close()
        return _scaler

########################################################################

def normalize(
        dataset: Union[TwoDDatasetBuilder, TwoDDatasetLoader],
        sub_dir: str = 'normalized',
        scaler_dict: Dict = {
            'node_attr': ['minmax_scaler', 0],
            'output': ['minmax_scaler', 0]
        }
) -> TwoDDatasetLoader:
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
        _variable_data = []
        for i in range(dataset.len()):
            _variable_data.append(dataset[i]._store[_variable])
        _variable_data = torch.cat(_variable_data, dim=0)
        if scaler_dict[_variable][1] is None:
            _variable_data = _variable_data.flatten().unsqueeze(1)
        elif len(_variable_data.shape) <= 1:
            _variable_data = _variable_data.unsqueeze(1)
        ###
        if scaler_dict[_variable][0] == 'minmax_scaler':
            _scaler = MinMaxScaler(feature_range=(0.,1.))
            _scaler.fit(_variable_data)
            _scaler_file = open(f'{dataset.root}/scaler/{_variable}.pkl','wb')
            pickle.dump(_scaler, _scaler_file)
            _scaler_file.close()
        ###
        if scaler_dict[_variable][0] == 'standard_scaler':
            _scaler = StandardScaler()
            _scaler.fit(_variable_data)
            _scaler_file = open(f'{dataset.root}/scaler/{_variable}.pkl','wb')
            pickle.dump(_scaler, _scaler_file)
            _scaler_file.close()
        ###
        if scaler_dict[_variable][0] == 'robust_scaler':
            _scaler = RobustScaler()
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
                _variable_data = torch.tensor(dataset.load_scaler(_variable).transform(_variable_data))
                _variable_data = torch.reshape(_variable_data, _resize)
            else:
                _variable_data = _data._store[_variable]
            setattr(_normalized_data, _variable, _variable_data)
        _file_name = dataset.processed_file_names()[i].replace(f'{dataset.root}','').replace(f'{dataset.sub}','').replace('/','')
        torch.save(_normalized_data, f'{dataset.root}/{sub_dir}/{_file_name}')
    
    return TwoDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)