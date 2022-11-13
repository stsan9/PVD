import os.path as osp
from model.refinement import GNN, FCN
import argparse
import torch
import numpy as np
import torch.nn.functional as F

def train(args):
    if args.model_type == 'GNN':
        model = GNN()
    elif args.model_type == 'FCN':
        model = FCN()

    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    gen_samples = torch.load(osp.join(args.data, 'samples.pth'))
    ground_truth = torch.load(osp.join(args.data, 'ref.pth'))

    dataset = torch.utils.data.TensorDataset(gen_samples, ground_truth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs)
    losses = []
    for epoch in range(args.epochs):
        _losses = []
        for gen_mb, gt_mb in dataloader:
            x = torch.tensor(gen_mb).to(torch.float32).cuda()
            y = torch.tensor(gt_mb).to(torch.float32).cuda()
    
            y_hat = model(x)
            
            loss = F.mse_loss(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            _losses.append(loss.item())
        losses.append(np.mean(_losses).item())
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}')
            print(f'Loss: {losses[-1]}')
            print()
    with open(f'{args.model_type}_losses.txt', 'w+') as f:
        f.write(str(losses))

    torch.save(model.state_dict(), args.model_type + '_refinement.pt')
        
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='GNN')
    parser.add_argument('--bs', default=256)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--patience', default=10)
    parser.add_argument('--epochs', default=10000)
    parser.add_argument('--data', default='/diffusionvol/experiments/test_generation/pvd_gluons/syn/', help='where sample.pth and ref.pth are stored')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # train loop)
    train(args)
