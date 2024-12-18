import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from typing import Optional, Callable, Union, List, Tuple
# from preprocessing import *
from data.graph_data import TorchGraphData
from sklearn.preprocessing import PowerTransformer
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from data.utils import *


########################################################################

class ThreeDDatasetBuilder(Dataset):
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
            ###
            _data = TorchGraphData()
            for _variable in _data_dict_input:
                setattr(_data, _variable, torch.tensor(_data_dict_input[_variable]).type(self.data_type))
            for _variable in _data_dict_output:
                setattr(_data, _variable, torch.tensor(_data_dict_output[_variable]).type(self.data_type))
            for _variable in _data_dict_sh:
                setattr(_data, _variable, torch.tensor(_data_dict_sh[_variable]).type(self.data_type))
            _data.edge_index = _data.edge_index.type(torch.LongTensor)
            ###   
            torch.save(_data, self.processed_file_names()[self.subjects.index(subject)])

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