import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import FloatTensor as FT
from tqdm import tqdm
from nngeometry.generator.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons


# training for 10 different seeds and then averaging over all runs
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# A two-layer neural netowrk with the following number of hidden units
hidden_dim = 500
n_samples = 300
epochs = 10000

experiments = [
    {'name': 'margin=-0.1',
     'dataset': lambda seed: two_moons_dataset(seed, margin=-0.1),
     'sp_coef': 0.0,
     'ls': ':'},  # ls = line style for plotting
    {'name': 'margin=+0.1',
     'dataset': lambda seed: two_moons_dataset(seed, margin=+0.1),
     'sp_coef': 0.0,
     'ls': '--'},
    {'name': 'margin=+0.1 with SP',
     'dataset': lambda seed: two_moons_dataset(seed, margin=+0.1),
     'sp_coef': 0.003,
     'ls': '-'}
     ]


# Network architecture
class Net(nn.Module):
    def __init__(self,
                 hidden_dim):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def two_moons_dataset(seed, margin=0.0):
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    y_pm1 = (2.0 * y - 1.0)[:, None]
    move_along_x = 0.5 * np.ones((n_samples, 1))
    move_along_y = y_pm1 * 0.3
    X = X - np.concatenate([move_along_x, move_along_y], 1)
    X[:, 1] = X[:, 1] - X[:, 1].mean()
    alter_sign = np.sign(-X[:, 1] * y_pm1[:, 0])
    X[:, 1] = X[:, 1] * alter_sign
    X[:, 1] = X[:, 1] - y_pm1[:, 0] * margin
    # Rotate data by 90 degrees
    X = X[:, ::-1].copy()
    return X, y


def NTK_Left_SV(net, X, y):
    def output_fn(input, target):
        # input = input.to('cuda')
        return net(input)

    layer_collection = LayerCollection.from_model(net)
    layer_collection.numel()
    batch = TensorDataset(X, y)
    batch_loader = DataLoader(batch)
    generator = Jacobian(layer_collection=layer_collection,
                         model=net,
                         loader=batch_loader,
                         function=output_fn,
                         n_output=1)
    jac = generator.get_jacobian()[0]
    K = torch.mm(jac, jac.transpose(0, 1))
    U, S, V = torch.svd(K, some=False)
    return U

# An n by n grid for the heatmap
n = 100
d1_min = -2
d1_max = 2
d2_min = -2
d2_max = 2
d1, d2 = torch.meshgrid([
    torch.linspace(d1_min, d1_max, n),
    torch.linspace(d2_min, d2_max, n)])
heatmap_plane = torch.stack((d1.flatten(), d2.flatten()), dim=1)
heatmap_avg = np.zeros((heatmap_plane.shape[0], len(experiments)))

# Z is the latent feature
Zs = np.zeros((epochs, len(experiments), len(seeds), n_samples))


for seed in seeds:
    print('Seed: ' + str(seed))

    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for exp_idx, exp in enumerate(experiments):
        print('Experiment: ' + exp['name'])
        X, y = exp['dataset'](seed)
        X, y = FT(X), FT(2.0 * y - 1.0)

        net = Net(hidden_dim)
        optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0002)
        U = NTK_Left_SV(net, X, y)

        for epoch in tqdm(range(epochs)):
            y_hat = net(X)[:, 0]
            loss = torch.log(1.0 + torch.exp(-y_hat * y)).mean()
            loss += exp['sp_coef'] * (y_hat ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Z = torch.mm(y_hat.unsqueeze(0), U)[0].data.cpu().numpy()
            Zs[epoch, exp_idx, seed] = np.abs(Z)
        print(loss.item())

        # Average heatmaps over seeds
        heatmap_avg[:, exp_idx] += net(heatmap_plane).data.cpu().numpy()[:, 0] / len(seeds)

# Average over seeds
Zs_mean = Zs.mean(axis=2)
Zs_std = Zs.std(axis=2)

# Plotting
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
cm = ListedColormap(['#C82506', '#0365C0'])
figure = plt.figure(figsize=((len(experiments) + 1) * 2.7, 3))
hmp_x = heatmap_plane[:, 0].data.numpy().reshape(n, n)
hmp_y = heatmap_plane[:, 1].data.numpy().reshape(n, n)

ax_last = plt.subplot(1, len(experiments) + 1, len(experiments) + 1)
for exp_idx, exp in enumerate(experiments):
    ax = plt.subplot(1, len(experiments) + 1, exp_idx + 1)
    # plot only one of the seeds
    X, y = exp['dataset'](seeds[0])
    hma = heatmap_avg[:, exp_idx].reshape(n, n)
    ax.contourf(hmp_x, hmp_y, hma, 10, cmap=plt.cm.RdBu, alpha=.8)
    ax.contour(hmp_x, hmp_y, hma, 10, antialiased=True, linewidths=0.2, colors='k')
    ax.contour(hmp_x, hmp_y, hma, 0, antialiased=True, linewidths=1.0, colors='k')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm, edgecolors='k', s=18)
    ax.axhline(y=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.axvline(x=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # We only plot Zs at index 1 and 6 out of n_samples
    ax_last.plot(Zs_mean[:, exp_idx, 1], color='#2D7F58', ls=exp['ls'], lw=1.5)
    ax_last.plot(Zs_mean[:, exp_idx, 6], color='#FA7F05', ls=exp['ls'], lw=1.5)
    ax_last.set_facecolor('#F9F9F9')
    # ax_last.plot()
    ax_last.grid(b=True, which='major', linestyle='--', lw=0.7, color='k', alpha=0.3)
    ax_last.set_yticklabels([])

plt.tight_layout()
plt.show()
