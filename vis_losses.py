import matplotlib.pyplot as plt
import argparse


def plot_losses(train_losses, val_losses):
    plt.plot(list(range(len(train_losses))), train_losses, label='Train Loss')
    plt.plot(list(range(len(val_losses))), val_losses, label='Val Loss')
    plt.title(f'{args.model_type} Refinement Network Losses')
    plt.xlabel('Epoch')
    if args.log_scale:
        plt.yscale('log')
    plt.ylabel(f'Average Loss{" (log scaled)" if args.log_scale else ""}')
    plt.legend()
    plt.savefig(f'{args.model_type}_vis.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', dest='model_type', type=str, default='GNN')
    parser.add_argument('--log_scale', action='store_true')
    args = parser.parse_args()

    train_filename = f'{args.model_type}_train_losses.txt'
    val_filename = f'{args.model_type}_val_losses.txt'

    train_losses, val_losses = [], []
    with open(train_filename, 'r') as f:
        for l in f:
            train_losses = eval(l.strip())
    with open(val_filename, 'r') as f:
        for l in f:
            val_losses = eval(l.strip())


    plot_losses(train_losses, val_losses)
