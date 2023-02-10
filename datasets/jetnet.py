import torch
import jetnet
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear

'''
custom_dataset
'''
class PointDataset(torch.utils.data.Dataset):
    def __init__(self, points, labels=None, generate=False) -> None:
        super(PointDataset, self).__init__()
        self.points = points
        self.labels = labels
        self.generate = generate

        if self.generate:
            self.m = points[..., :-1].mean(axis=1).reshape(-1, 1, 3)
            self.std = points[..., :-1].std(axis=1).reshape(-1, 1, 3)
        
    def __getitem__(self, idx : int) -> torch.tensor:
        current_points = self.points[idx]
        current_points = torch.from_numpy(current_points).float()

        labels = None
        if self.labels is not None:
            labels = self.labels[idx]

        if self.generate:
            m = self.m[idx]
            std = self.std[idx]
            m = torch.from_numpy(m).float()
            std = torch.from_numpy(std).float()
        
            return {
                'train_points': current_points,
                'test_points': current_points,  # doesn't really matter
                'idx': idx,
                'mean': m,
                'std': std,
                'labels': labels   # jet features
            }
        else:
            return {
                'train_points': current_points,
                'idx': idx,
                'labels': labels   # jet features
            }

    def __len__(self) -> int:
        return len(self.points)

def load_gluon_dataset(dataroot, dataset_size=1000, generate=False):
    particle_data, jet_data = JetNet.getData(jet_type=["g"], data_dir=dataroot)
    particle_data = particle_data
    np.random.shuffle(particle_data)
    particle_data = particle_data[:dataset_size]
    return PointDataset(particle_data, jet_data, generate=generate)