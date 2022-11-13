import matplotlib.pyplot as plt
import argparse


def plot_losses(losses):
    plt.plot(list(range(len(losses))), losses)
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.yscale('log')
    plt.savefig(f'{args.filename.split(".")[0]}_vis.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', dest='filename', type=str, default=None, required=True)
    args = parser.parse_args()

    losses = []
    with open(args.filename, 'r') as f:
        for l in f:
            losses = eval(l.strip())


    plot_losses(losses)
