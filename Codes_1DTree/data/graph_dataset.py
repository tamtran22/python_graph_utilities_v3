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

#####################################################
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
        refined_max_length: float = 5.,
        data_type = torch.float64,
        readme=None
    ):
        ##
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self.sub = sub_dir
        self.refined_max_length = refined_max_length
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
            data_dict = read_1D_input(_input_file_name)

            _output_file_names = [file_name_output(subject, time) for time in self.time_names]
            data_dict.update(read_1D_output(_output_file_names))
            
            data_dict = process_data(data_dict, max_length=self.refined_max_length)

            _shellscript_file_name = file_name_sh(subject)
            data_dict.update(read_shell_script(_shellscript_file_name))

            ###
            edge_index = torch.tensor(
                data_dict['edge_index']
            ).type(torch.LongTensor)

            edge_index_raw = torch.tensor(
                data_dict['edge_index_raw']
            ).type(torch.LongTensor)

            original_flag = torch.tensor(
                data_dict['original_flag']
            ).type(torch.LongTensor)

            node_attr = torch.tensor(np.concatenate([
                data_dict['coordinate'],
            ], axis=1)).type(self.data_type)

            edge_attr = torch.tensor(np.concatenate([
                np.expand_dims(data_dict['length'], axis=1),
                np.expand_dims(data_dict['diameter'], axis=1),
                np.expand_dims(data_dict['generation'], axis=1),
                np.expand_dims(data_dict['lobe'], axis=1),
                np.expand_dims(data_dict['flag'], axis=1)
            ], axis=1)).type(self.data_type)

            global_attr = torch.tensor(np.concatenate([
                np.expand_dims(data_dict['global_attr'], axis=0)
            ]*node_attr.size(0), axis=0))

            pressure_attr = torch.tensor(data_dict['pressure']).type(self.data_type)
            reference_pressure = pressure_attr[0][0].item()
            pressure_attr -= reference_pressure

            flowrate_attr = torch.tensor(data_dict['flowrate']).type(self.data_type)

            data = TorchGraphData()
            setattr(data, 'edge_index', edge_index)
            setattr(data, 'edge_index_raw', edge_index_raw)
            setattr(data, 'original_flag', original_flag)
            setattr(data, 'node_attr', torch.cat([node_attr, global_attr], dim=1))
            setattr(data, 'edge_attr', edge_attr)
            setattr(data, 'pressure', pressure_attr)
            setattr(data, 'flowrate', flowrate_attr)

            torch.save(data, self.processed_file_names()[self.subjects.index(subject)])
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
        processed_file_names =  os.listdir(f'{self.root}/{self.sub}/')
        # print(processed_file_names)
        # reduced_processed_file_names = [file_name.replace(self.root,'').replace(self.sub,'').replace('/','') for file_name in processed_file_names]
        # print(reduced_processed_file_names)
        processed_file_names = [f'{self.root}/{self.sub}/{file_name}' for file_name in processed_file_names]
        return processed_file_names

    def len(self):
        return len(self.processed_file_names())

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