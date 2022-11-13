import os.path as osp
from model.refinement import GNN, FCN
import argparse
import torch

def train(args):
    if args.model_type == 'GNN':
        model = GNN()
    elif args.model_type == 'FCN':
        model = FCN()
    
    gen_samples = torch.load(osp.join(args.data, 'sample.pth'))
    ground_truth = torch.load(osp.join(args.data, 'ref.pth'))

    dataset = torch.utils.data.TensorDataset(gen_samples, ground_truth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs)





def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='GNN')
    parser.add_argument('--bs', default=256)
    parser.add_argument('--patience', default=10)
    parser.add_argument('--epochs', default=10000)
    parser.add_argument('--data', default='/diffusionvol/experiments/test_generation/pvd_gluons/syn/', help='where sample.pth and ref.pth are stored')

if __name__ == '__main__':
    # train loop)
    train(args)