import os.path as osp
from model.refinement import GNN, FCN
from model.mpgan.model import MPNet
import argparse
import torch
import numpy as np
import torch.nn.functional as F

def eval_loss(dataloader, model):
    _losses = []
    with torch.no_grad():
        for x, y in dataloader:
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            _losses.append(loss.item())
    return np.mean(_losses).item()
    

def train(args):
    if args.model_type == 'GNN':
        model = GNN()
    elif args.model_type == 'FCN':
        model = FCN()
    elif args.model_type == 'MPNet':
        model = MPNet(30, 3, output_node_size=3)

    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load entire x's and y's
    gen_samples = torch.load(osp.join(args.data, 'samples.pth'))
    ground_truth = torch.load(osp.join(args.data, 'ref.pth'))

    # Get train, val, test splits
    data = torch.cat((gen_samples, ground_truth), dim=2)
    n = gen_samples.shape[0]
    splits = [int(0.70*n), int(0.15*n), int(0.15*n)]
    train_set, val_set, test_set = torch.utils.data.random_split(data, splits, generator=torch.Generator().manual_seed(1))
    train_x, train_y = train_set.dataset[train_set.indices][..., :3], train_set.dataset[train_set.indices][..., 3:]
    val_x, val_y = val_set.dataset[val_set.indices][..., :3], val_set.dataset[val_set.indices][..., 3:]
    test_x, test_y = test_set.dataset[test_set.indices][..., :3], test_set.dataset[test_set.indices][..., 3:]

    # Put val and test data on cuda
    train_x, train_y = train_x.cuda(), train_y.cuda()
    val_x, val_y = val_x.cuda(), val_y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()

    # Define train dataloader for minibatch training
    train_set = torch.utils.data.TensorDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs)

    # Since we can't fit all val and test data to cuda memory, use dataloaders too
    val_set = torch.utils.data.TensorDataset(val_x, val_y)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.bs)
    test_set = torch.utils.data.TensorDataset(test_x, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs)

    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        _train_losses = []
        for x, y in train_dataloader:
    
            y_hat = model(x)
            
            loss = F.mse_loss(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            _train_losses.append(loss.item())

        train_loss = np.mean(_train_losses).item()
        train_losses.append(train_loss)

        val_loss = eval_loss(val_dataloader, model)
        val_losses.append(val_loss)

        if len(val_losses) > args.patience and val_loss >= np.max(val_losses[-args.patience:]):
            print(f'Early stopping at epoch {epoch}.')
            break

        if epoch % 100 == 0:

            print(f'Epoch: {epoch}')
            print(f'Train Loss: {train_losses[-1]}')
            print(f'Val Loss: {val_losses[-1]}')
            print()

    with open(f'./refinement/data/{args.model_type}_train_losses.txt', 'w+') as f:
        f.write(str(train_losses))
    with open(f'./refinement/data/{args.model_type}_val_losses.txt', 'w+') as f:
        f.write(str(val_losses))

    torch.save(model.state_dict(), './refinement/models/' + args.model_type + '_refinement.pt')

    test_loss = eval_loss(val_dataloader, model)
    print(f'Test Loss: {test_loss}')
    with open(f'./refinement/data/{args.model_type}_test_loss.txt', 'w+') as f:
        f.write(str(test_loss) + '\n')
        
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='GNN')
    parser.add_argument('--bs', default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', default=10000)
    parser.add_argument('--data', default='/diffusionvol/experiments/test_generation/pvd_gluons/syn/', help='where sample.pth and ref.pth are stored')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # train loop)
    train(args)
