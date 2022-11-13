import torch
import torch.nn as nn
import numpy as np
import itertools
from torch_geometric.nn import EdgeConv

class GNN(nn.Module):
    def __init__(self, input_dim=3, h_dim=16, n_particles=30):
        super(GNN, self).__init__()
        refinement_nn = nn.Sequential(
            nn.Linear(2 * input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim)
        )
        self.gcn = EdgeConv(nn=refinement_nn, aggr='mean')

        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
        # fully connected graph
        self.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    

    def forward(self, x):
        # x : [N, 30, 3]
        x = self.gcn(x, self.edge_index)
        return x