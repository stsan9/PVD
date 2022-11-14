
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch_geometric.nn import EdgeConv

class GNN(nn.Module):
    def __init__(self, input_dim=3, h_dim=32, n_particles=30, device=torch.device('cuda')):
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
        self.gcn = EdgeConv(nn=refinement_nn, aggr='mean').cuda()

        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
        # fully connected graph
        self.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous().cuda()
    

    def forward(self, x):
        # x : [N, 30, 3]
        x = self.gcn(x, self.edge_index)
        return x

class GNN2(nn.Module):
    def __init__(self, input_dim=3, h_dim=32, n_particles=30, device=torch.device('cuda')):
        super(GNN2, self).__init__()
        refinement_nn = nn.Sequential(
            nn.Linear(2 * input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        refinement_nn_2 = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim)
        )
        self.gcn1 = EdgeConv(nn=refinement_nn, aggr='mean').cuda()
        self.gcn2 = EdgeConv(nn=refinement_nn_2, aggr='mean').cuda()

        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
        # fully connected graph
        self.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous().cuda()
    

    def forward(self, x):
        # x : [N, 30, 3]
        x = self.gcn1(x, self.edge_index)
        x = self.gcn2(x, self.edge_index)
        return x

class NodeFCN(nn.Module):
    def __init__(self, input_dim=3, h_dim=32, n_particles=30, device=torch.device('cuda')):
        super(NodeFCN, self).__init__()
        refinement_nn = nn.Sequential(
            nn.Linear(2 * input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim)
        )
        self.gcn = EdgeConv(nn=refinement_nn, aggr='mean').cuda()

        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m==n])
        # unconnected graph
        self.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous().cuda()
    

    def forward(self, x):
        # x : [N, 30, 3]
        x = self.gcn(x, self.edge_index)
        return x

class NodeFCN2(nn.Module):
    def __init__(self, input_dim=3, h_dim=32, n_particles=30, device=torch.device('cuda')):
        super(NodeFCN2, self).__init__()
        refinement_nn = nn.Sequential(
            nn.Linear(2 * input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        refinement_nn_2 = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim)
        )
        self.gcn1 = EdgeConv(nn=refinement_nn, aggr='mean').cuda()
        self.gcn2 = EdgeConv(nn=refinement_nn_2, aggr='mean').cuda()

        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m==n])
        # unconnected graph
        self.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous().cuda()
    

    def forward(self, x):
        # x : [N, 30, 3]
        x = self.gcn1(x, self.edge_index)
        x = self.gcn2(x, self.edge_index)
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
                    nn.Linear(p*c, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, p*c)
                ).cuda()

    def forward(self, x):
        """
            x shape: (n, p, c)
        """
        return self.model(torch.flatten(x, start_dim=1, end_dim=-1)).reshape(x.shape[0], self.p, self.c)
