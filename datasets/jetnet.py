import torch
import jetnet
import numpy as np
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear

'''
custom_dataset
'''
class PointDataset(torch.utils.data.Dataset):
    def __init__(self, points, labels=None) -> None:
        super(PointDataset, self).__init__()
        self.points = points
        self.labels = labels
        
    def __getitem__(self, idx : int) -> torch.tensor:
        current_points = self.points[idx]
        labels = None
        if self.labels is not None:
            labels = self.labels[idx]
        current_points = torch.from_numpy(current_points).float()
        return {
            'train_points': current_points,
            'idx': idx,
            'labels': labels   # jet features
        }
    
    def __len__(self) -> int:
        return len(self.points)

def load_gluon_dataset(dataroot, dataset_size=1000):
    particle_data, jet_data = JetNet.getData(jet_type=["g"], data_dir=dataroot)
    particle_data = particle_data
    np.random.shuffle(particle_data)
    particle_data = particle_data[:dataset_size]
    return PointDataset(particle_data, jet_data)
