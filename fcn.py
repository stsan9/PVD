class FCN(nn.Module):
    '''
    Fully connected refine network
    '''
    def __init__(self, p, c):
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
                    nn.Linear(1024, p*c)
                )

    def forward(self, x):
        """
            x shape: (n, p, c)
        """
        return self.model(torch.flatten(x, start_dim=1, end_dim=-1)).reshape(x.shape[0], self.p, self.c)
