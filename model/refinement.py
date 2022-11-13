
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

class FCN(nn.Module):
    '''
    Fully connected refine network
    '''
    def __init__(self, p=30, c=3):
        """
            args:
                p: # particles, expected 30
                c: # features per particle, expected 3
        """
        super(FCN, self).__init__()
        self.p = p
        self.c = c
        self.model = nn.Sequential(
                    nn.Linear(p*c, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, p*c)
                )

    def forward(self, x):
        """
            x shape: (n, p, c)
        """
        return self.model(torch.flatten(x, start_dim=1, end_dim=-1)).reshape(x.shape[0], self.p, self.c)
