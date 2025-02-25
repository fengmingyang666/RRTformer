import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RRTStarDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path, allow_pickle=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tree_nodes, next_sample_point, env_map = self.data[idx]
        tree_nodes_tensor = torch.tensor(tree_nodes, dtype=torch.float32)
        next_sample_point_tensor = torch.tensor(next_sample_point, dtype=torch.float32)
        env_map_tensor = torch.tensor(env_map, dtype=torch.float32)
        return tree_nodes_tensor, next_sample_point_tensor, env_map_tensor

def collate_fn(batch):
    tree_nodes_batch = [item[0] for item in batch]
    next_sample_points_batch = [item[1] for item in batch]
    env_maps_batch = [item[2] for item in batch]
    padded_tree_nodes = pad_sequence(tree_nodes_batch, batch_first=True, padding_value=0.0)
    next_sample_points = torch.stack(next_sample_points_batch)
    env_maps = torch.stack(env_maps_batch)
    return padded_tree_nodes, next_sample_points, env_maps

def get_data_loader(dataset_path, batch_size=32, shuffle=True):
    dataset = RRTStarDataset(dataset_path)
    print("DataSet Size: ", dataset.data.shape)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader
