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
from data.graph_dataset import OneDDatasetBuilder, OneDDatasetLoader


################################################################################
def normalize(
    dataset: Union[OneDDatasetLoader, OneDDatasetBuilder],
    sub_dir: str = 'normalized',
    scaler_dict: Dict = {
        'node_attr': ['minmax_scaler', 'minmax_scaler'],
        # 'pressure' : 'minmax_scaler'
    },
    clipping: float = 1e-3,
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
    for variable in scaler_dict:

        variable_data = []
        for i in range(dataset.len()):
            variable_data.append(dataset[i]._store[variable])
        variable_data = torch.cat(variable_data, dim=0)
        
        if isinstance(scaler_dict[variable], str):
            variable_data = variable_data.flatten().unsqueeze(1)
            scaler = resolve_scaler(scaler_dict[variable], feature_range=(0,1))
            variable_min = np.quantile(variable_data.numpy(), q=clipping)
            variable_max = np.quantile(variable_data.numpy(), q=1-clipping)
            variable_data = torch.clip(variable_data, variable_min, variable_max)
            scaler.fit(variable_data)
            scaler_file = open(f'{dataset.root}/scaler/{variable}.pkl', 'wb')
            pickle.dump(scaler, scaler_file)
            scaler_file.close()
        
        if isinstance(scaler_dict[variable], List):
            scaler = []
            for i in range(variable_data.size(1)):
                _scaler = resolve_scaler(scaler_dict[variable][i], feature_range=(0,1))
                _scaler.fit(variable_data[:, i].unsqueeze(1))
                scaler.append(_scaler)
            scaler_file = open(f'{dataset.root}/scaler/{variable}.pkl','wb')
            pickle.dump(scaler, scaler_file)
            scaler_file.close()
    ##
    for i in range(dataset.len()):
        data = dataset[i]
        normalized_data = TorchGraphData()
        for variable in data._store:
            variable_data = data._store[variable]

            if not variable in scaler_dict:
                setattr(normalized_data, variable, variable_data)
                continue
            
            if isinstance(scaler_dict[variable], str):
                size = variable_data.size()
                variable_data = variable_data.flatten().unsqueeze(1)
                scaler = pickle.load(open(f'{dataset.root}/scaler/{variable}.pkl', 'rb'))
                variable_data = torch.tensor(scaler.transform(variable_data))
                variable_data = torch.reshape(variable_data, size)
                setattr(normalized_data, variable, variable_data)
                continue

            if isinstance(scaler_dict[variable], List):
                setattr(normalized_data, variable, variable_data)
                scaler = pickle.load(open(f'{dataset.root}/scaler/{variable}.pkl', 'rb'))
                normalize_variable_data = []
                for j in range(variable_data.size(1)):
                    _variable_data = variable_data[:, j].unsqueeze(1)
                    _scaler = scaler[j]
                    _variable_data = torch.tensor(_scaler.transform(_variable_data))
                    normalize_variable_data.append(_variable_data)
                variable_data = torch.cat(normalize_variable_data, dim=1)
                setattr(normalized_data, variable, variable_data)
                continue
        
        file_name = dataset.processed_file_names()[i].replace(f'{dataset.root}','').replace(f'{dataset.sub}','').replace('/','')
        torch.save(normalized_data, f'{dataset.root}/{sub_dir}/{file_name}')

    return OneDDatasetLoader(root_dir=dataset.root, sub_dir=sub_dir)

###################################################################
def resolve_scaler(scaler_type, **kwargs):
    if scaler_type == 'minmax_scaler':
        return MinMaxScaler(**kwargs)
    if scaler_type == 'standard_scaler':
        return StandardScaler()
    if scaler_type == 'robust_scaler':
        return RobustScaler()
    
###################################################################
def batchgraph_generation_wise(
        dataset: Union[OneDDatasetBuilder, OneDDatasetLoader],
        sub_dir: str = 'batched',
        batch_gens: List[Tuple] = [[0,9],[10,13], [14, 15], [16, 18], [18,50]],
        subset_hops: int = 1,
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
            _gen = dataset[i].edge_attr[:,2].numpy().astype(np.int32)
            _gen, _ = edge_to_node(_gen, dataset[i].edge_index.numpy())
        else:
            _scaler = dataset.load_scaler('edge_attr')[2]
            _edge_attr = _scaler.inverse_transform(dataset[i].edge_attr)
            _gen = _edge_attr[:,2].astype(np.int32)
            _gen, _  = edge_to_node(_gen, dataset[i].edge_index.numpy())
        _batch_gens = []
        for batch_gen in batch_gens:
            _batch_gen = [list(range(batch_gen[0], batch_gen[1]))]
            _batch_gens += _batch_gen
        _subsets = [list(np.where(np.isin(_gen, _batch_gens[j]))[0]) for j in range(len(batch_gens))]
        ###
        _subgraphs = get_batchgraphs(
            data=dataset[i],
            subsets=_subsets,
            subset_size=1,
            subset_hops=subset_hops,
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
        for _data_subset in data_subset_dict:
            data_subset_dict[_data_subset] = list(np.where(np.isin(_batching_id, data_subset_dict[_data_subset] ) == True)[0])
    _data_subsets = []
    for _data_subset in data_subset_dict:
        _data_subsets.append(DataLoader([dataset[i] for i in data_subset_dict[_data_subset]], batch_size=n_data_per_batch))
    return tuple(_data_subsets)